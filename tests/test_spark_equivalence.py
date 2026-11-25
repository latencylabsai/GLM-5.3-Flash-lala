"""The release check must reject token mismatches and missing benchmark samples."""

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("spark_outputs", ROOT / "benchmarks/check_spark_outputs.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class SparkEquivalenceTests(unittest.TestCase):
    def records(self, directory, mismatch=False, incomplete=False):
        for mode in ("serial", "mtp"):
            rows = []
            for i in range(6):
                ids = list(range(256 if i < 3 else 128))
                if mismatch and mode == "mtp" and i == 4:
                    ids[17] += 1
                rows.append(json.dumps({
                    "token_ids": ids,
                    "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "accepted_draft_tokens": 20 if mode == "mtp" else 0,
                    "rejected_draft_tokens": 3 if mode == "mtp" else 0,
                }))
            if incomplete and mode == "mtp":
                rows.pop()
            (directory / f"{mode}-tokens.jsonl").write_text("\n".join(rows) + "\n")

    def test_exact_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.records(root)
            self.assertTrue(checker.compare(root)["exact_output_token_agreement"])

    def test_mismatch_and_missing_sample_fail(self):
        for kwargs in ({"mismatch": True}, {"incomplete": True}):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                self.records(root, **kwargs)
                with self.assertRaises(AssertionError):
                    checker.compare(root)
