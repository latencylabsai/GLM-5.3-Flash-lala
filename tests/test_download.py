"""Download integrity and interruption checks without network or model weights."""

import contextlib
import hashlib
import io
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm_lala.cli import download


class DownloadIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = pathlib.Path(self.temporary.name) / "snapshot"
        self.files = {
            "config.json": b"{}",
            "model.safetensors.index.json": json.dumps(
                {"weight_map": {"test_tensor": "weights.safetensors"}}
            ).encode(),
            "weights.safetensors": b"test weight bytes",
        }
        entries = []
        for name, data in self.files.items():
            lfs = name.endswith(".safetensors")
            entries.append(
                SimpleNamespace(
                    rfilename=name,
                    size=len(data),
                    lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
                    if lfs
                    else None,
                    blob_id=hashlib.sha1(
                        f"blob {len(data)}\0".encode() + data
                    ).hexdigest(),
                )
            )
        self.info = SimpleNamespace(sha="test-revision", siblings=entries)
        self.args = SimpleNamespace(
            output=self.output, revision=self.info.sha, metadata_only=False
        )

    def write_snapshot(self, *_args, **_kwargs):
        for name, data in self.files.items():
            (self.output / name).write_bytes(data)

    def run_download(self, writer=None):
        with (
            patch("huggingface_hub.HfApi") as api,
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=writer or self.write_snapshot,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            api.return_value.model_info.return_value = self.info
            download(self.args)

    def test_full_download_records_verified_files(self):
        self.run_download()
        receipt = json.loads((self.output / "lala-download.json").read_text())
        self.assertTrue(receipt["all_selected_files_verified"])
        self.assertEqual(
            {entry["file"] for entry in receipt["verified_files"]}, set(self.files)
        )

    def test_same_revision_resumes_after_partial_transfer(self):
        def interrupted(*_args, **_kwargs):
            (self.output / "config.json").write_bytes(self.files["config.json"])
            raise ConnectionError("Interrupted transfer")

        with self.assertRaises(ConnectionError):
            self.run_download(interrupted)
        self.assertFalse((self.output / "lala-download.json").exists())
        self.run_download()
        self.assertTrue((self.output / "lala-download.json").is_file())

    def test_different_revision_cannot_reuse_destination(self):
        self.run_download()
        self.info.sha = "other-revision"
        with self.assertRaisesRegex(ValueError, "empty output directory"):
            self.run_download()

    def test_corrupt_weights_or_metadata_cannot_receive_success_receipt(self):
        for name in ("config.json", "weights.safetensors"):
            with self.subTest(file=name):

                def corrupted(*_args, file_name=name, **_kwargs):
                    self.write_snapshot()
                    data = self.files[file_name]
                    (self.output / file_name).write_bytes(
                        bytes([data[0] ^ 1]) + data[1:]
                    )

                with self.assertRaisesRegex(ValueError, "checksum differs"):
                    self.run_download(corrupted)
                self.assertFalse((self.output / "lala-download.json").exists())

    def test_failed_reverification_invalidates_prior_receipt(self):
        self.run_download()

        def corrupted(*_args, **_kwargs):
            (self.output / "config.json").write_bytes(b"[]")

        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.run_download(corrupted)
        self.assertFalse((self.output / "lala-download.json").exists())


if __name__ == "__main__":
    unittest.main()
