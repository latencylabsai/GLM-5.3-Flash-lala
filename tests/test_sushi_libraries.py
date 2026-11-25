"""Release staging rejects corrupted downloads and preserves existing runtimes."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "stage", Path(__file__).resolve().parents[1] / "scripts/stage-sushi-libraries.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


class SushiLibrariesTests(unittest.TestCase):
    def fixture(self, root):
        archive = root / "release.tar.gz"
        files = {"guest.json": json.dumps({"version": "1.7.3", "commit": "source"}).encode(),
                 "LICENSE": b"license", "LICENSE-APACHE-2.0": b"license", "NOTICE": b"notice"}
        files.update({"lib/" + name: name.encode() for name in stage.LIBRARIES})
        with tarfile.open(archive, "w:gz") as bundle:
            for name, data in files.items():
                member = tarfile.TarInfo("sushi-macos-arm64/" + name)
                member.size = len(data)
                bundle.addfile(member, io.BytesIO(data))
        pin = {"version": "1.7.3", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        mac = {"revision": "source", "mlx": "mlx-pin", "mlx_c": "mlxc-pin"}
        return archive, pin, mac

    def test_verified_stage_repeat_and_local_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, pin, mac = self.fixture(root)
            dest = root / "mlx"
            stage.stage(archive, dest, pin, mac)
            stage.stage(archive, dest, pin, mac)
            (dest / "lib/libmlx.dylib").write_bytes(b"local change")
            with self.assertRaisesRegex(ValueError, "Existing MLX stage differs"):
                stage.stage(archive, dest, pin, mac)
            self.assertEqual((dest / "lib/libmlx.dylib").read_bytes(), b"local change")

    def test_bad_hash_or_metadata_does_not_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, pin, mac = self.fixture(root)
            for candidate, source in (({**pin, "sha256": "wrong"}, mac),
                                      (pin, {**mac, "revision": "wrong"})):
                with self.assertRaises(ValueError):
                    stage.stage(archive, root / "mlx", candidate, source)
                self.assertFalse((root / "mlx").exists())
