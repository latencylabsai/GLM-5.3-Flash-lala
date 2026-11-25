"""Verify default Spark native MTP and serial config without loading weights."""

import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm_lala.cli import spark_command


class SparkDefaultsTests(unittest.TestCase):
    def command(self, serial=False, mode=None):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "quantization_config.json").write_text("{}")
            python = root / ".runtime/spark-venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            args = SimpleNamespace(
                model=root,
                mtp_pack=None,
                serial=serial,
                spark_draft=mode,
                context=262144,
                host="127.0.0.1",
                port=8000,
                max_tokens=4096,
            )
            with (
                patch("platform.system", return_value="Linux"),
                patch("platform.machine", return_value="aarch64"),
                patch("glm_lala.cli.ROOT", root),
                patch.dict(os.environ, {"LALA_SUFFIX_TOKENS": "11"}),
            ):
                command, env = spark_command(args)
                self.assertNotIn("LALA_SUFFIX_TOKENS", env)
                self.assertEqual(env["TABBY_DIR"], str(root / ".runtime/tabby-native"))
            return command, json.loads((root / ".runtime/spark-8000.json").read_text())

    def test_default_mtp_and_vision(self):
        command, config = self.command()
        self.assertEqual(config["draft_model"]["draft_mode"], "mtp")
        self.assertEqual(config["draft_model"]["draft_num_tokens"], 4)
        self.assertEqual(config["draft_model"]["hybrid_draft_tokens"], 0)
        self.assertTrue(config["model"]["vision"])
        self.assertEqual(config["model"]["max_seq_len"], 262144)
        self.assertIn("glm_lala.spark_server", command)
        self.assertNotIn("docker", command)
        self.assertEqual(config["network"]["host"], "127.0.0.1")
        self.assertEqual(config["network"]["port"], 8000)

    def test_explicit_serial_and_conflicting_flags(self):
        for serial, mode in [(True, None), (False, "serial")]:
            _, config = self.command(serial, mode)
            self.assertEqual(config["draft_model"]["draft_mode"], "disabled")
            self.assertTrue(config["model"]["vision"])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.command(True, "mtp")
