"""Standalone native MTP must never discover or load a DFlash2 assistant."""

import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm_lala.cli import mac_command


class MacMtpLaunchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model = pathlib.Path(directory.name)
        (self.model / "config.json").write_text(
            json.dumps({"sushi_pack": {"format": "glm5-next-hf-names-1"}})
        )
        self.pack = self.model / "mtp.safetensors"
        self.pack.touch()

    def command(self, pack=None, serial=False):
        args = SimpleNamespace(
            model=self.model,
            serial=serial,
            binary=None,
            host="127.0.0.1",
            port=8000,
            context=262144,
            max_tokens=4096,
            mtp_pack=pack,
        )
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
        ):
            return mac_command(args)

    def test_native_head_without_assistant_or_suffix(self):
        command, env = self.command(self.pack)
        self.assertIn("--no-drafter", command)
        self.assertNotIn("--drafter", command)
        self.assertIn("--vision", command)
        self.assertIn("--glm-native-mtp", command)
        self.assertIn("--no-mtp-lookup", command)
        self.assertEqual(env["SUSHI_GLM_MTP_PACK"], str(self.pack.resolve()))
        self.assertEqual(env["SUSHI_GLM_MTP_HISTORY"], "1")

    def test_serial_drops_inherited_mtp_state(self):
        with patch.dict(
            os.environ, {"SUSHI_GLM_MTP_PACK": "/unwanted/mtp.safetensors"}
        ):
            command, env = self.command(serial=True)
        self.assertIn("--no-drafter", command)
        self.assertNotIn("SUSHI_GLM_MTP_PACK", env)

    def test_serial_rejects_mtp_options(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.command(self.pack, serial=True)

    def test_default_resolves_sibling_pack_and_missing_pack_fails(self):
        with self.assertRaisesRegex(ValueError, "enabled by default"):
            self.command()
        pack_dir = self.model / "mtp"
        pack_dir.mkdir()
        (pack_dir / "mtp.safetensors").touch()
        self.model = self.model / "mac"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"sushi_pack": true}')
        command, env = self.command()
        self.assertIn("--glm-native-mtp", command)
        self.assertIn("--vision", command)
        self.assertEqual(
            env["SUSHI_GLM_MTP_PACK"], str((pack_dir / "mtp.safetensors").resolve())
        )
