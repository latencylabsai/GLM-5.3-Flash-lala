"""Removed experimental paths must fail before loading a runtime."""
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from glm_lala.cli import main


class RemovedModesTests(unittest.TestCase):
    def test_removed_options_are_rejected(self):
        for options in (
            ["--backend", "mac", "--mtp-suffix"],
            ["--backend", "spark", "--spark-draft", "suffix"],
            ["--backend", "spark", "--spark-runtime", "docker"],
            ["--backend", "spark", "--image", "old-image"],
        ):
            with self.subTest(options=options), patch.object(
                sys, "argv", ["glm-lala", "serve", "--model", "/unused", *options]
            ), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
