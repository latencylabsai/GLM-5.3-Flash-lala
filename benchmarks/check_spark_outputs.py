"""Require exact emitted-token agreement for the paired Spark 4K benchmark."""

import argparse
import hashlib
import json
from pathlib import Path


def compare(directory):
    records = {}
    for mode in ("serial", "mtp"):
        rows = [json.loads(line) for line in (directory / f"{mode}-tokens.jsonl").read_text().splitlines()]
        assert len(rows) == 6, f"Expected six completed {mode} requests, got {len(rows)}"
        for i, row in enumerate(rows):
            assert len(row["token_ids"]) == (256 if i < 3 else 128)
            assert row["sha256"] == hashlib.sha256(json.dumps(row["token_ids"]).encode()).hexdigest()
        records[mode] = rows
    samples = []
    for i, (serial, mtp) in enumerate(zip(records["serial"], records["mtp"])):
        assert serial["token_ids"] == mtp["token_ids"], f"Token mismatch in sample {i + 1}"
        samples.append({
            "workload": "code" if i < 3 else "prose",
            "sample": i % 3 + 1,
            "output_tokens": len(mtp["token_ids"]),
            "output_ids_sha256": mtp["sha256"],
            "accepted_draft_tokens": mtp["accepted_draft_tokens"],
            "rejected_draft_tokens": mtp["rejected_draft_tokens"],
        })
    assert sum(s["accepted_draft_tokens"] for s in samples) > 0
    assert sum(s["rejected_draft_tokens"] for s in samples) > 0
    return {"passed": True, "exact_output_token_agreement": True, "samples": samples}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = compare(args.directory)
    (args.directory / "equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    print("All six MTP outputs match serial token for token; acceptance and rejection exercised.")
