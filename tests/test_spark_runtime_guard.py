"""An old CUDA extension must fail before loading a model."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from glm_lala import spark_server


class SparkRuntimeGuardTests(unittest.TestCase):
    def test_unpatched_binary_is_rejected(self):
        modules = {
            "mixedk": SimpleNamespace(),
            "exllamav3_ext": SimpleNamespace(),
            "exllamav3": SimpleNamespace(),
            "exllamav3.version": SimpleNamespace(__version__="1.6.2"),
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, clear=True):
            with self.assertRaisesRegex(RuntimeError, "missing the Lala decode correction"):
                spark_server.main()
            self.assertEqual(os.environ["LALA_EXL3_CONSISTENT"], "1")
            self.assertEqual(os.environ["EXL3_INT8_GEMV"], "0")
            self.assertEqual(os.environ["EXL3_MOE_COOP_WIDE"], "1")
            self.assertEqual(os.environ["EXL3_MOE_COOP_KSPLIT"], "1")
