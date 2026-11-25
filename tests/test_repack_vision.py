"""Vision banks must be converted even when absent from native metadata."""

import unittest

from glm_lala.repack import compressed_keys


class VisionLayoutTests(unittest.TestCase):
    def test_vision_bank_missing_from_manifest_is_discovered(self):
        name = "model.visual.blocks.0.attn.k_proj"
        index = {
            f"{name}.{part}": "weights.safetensors"
            for part in ("trellis", "suh", "svh", "mul1")
        }
        self.assertEqual(
            compressed_keys({}, index),
            {f"{name}.{part}" for part in ("trellis", "suh", "svh")},
        )

    def test_incomplete_native_bank_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete native EXL3"):
            compressed_keys({}, {"model.visual.test.trellis": "weights.safetensors"})
