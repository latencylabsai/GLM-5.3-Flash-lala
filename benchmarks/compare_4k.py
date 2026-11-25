"""Replay Sushi's captured 4K code/prose token sequences on either API."""

import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import time

from validate import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:43182")
    parser.add_argument("--backend", choices=["mac", "spark"], required=True)
    parser.add_argument("--mode", choices=["serial", "accelerated"], required=True)
    parser.add_argument("--fixtures", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    model = request(args.url, "/v1/models")["data"][0]["id"]
    record = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "backend": args.backend,
        "mode": args.mode,
        "model": model,
        "vision_enabled": True,
        "context_capacity": 262144,
        "kv_precision": "16-bit",
        "greedy": True,
        "ignore_eos": True,
        "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "arms": [],
    }
    for fixture in fixtures["arms"]:
        ids = fixture["prompt"]
        assert len(ids) == 4096
        prompt = ids
        if args.backend == "spark":
            prompt = request(args.url, "/v1/token/decode", {"tokens": ids})["text"]
            actual = request(args.url, "/v1/token/encode", {"text": prompt})["tokens"]
            if actual != ids:
                raise ValueError(
                    "Spark tokenizer does not reproduce the exact Sushi prompt IDs"
                )
        arm = {
            "name": fixture["name"],
            "prompt_tokens": len(ids),
            "prompt_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "output_tokens": fixture["max_tokens"],
            "samples": [],
        }
        for sample in range(3):
            body = {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
                "max_tokens": fixture["max_tokens"],
            }
            if args.backend == "mac":
                body["enable_drafter"] = args.mode == "accelerated"
            start = time.monotonic()
            response = request(args.url, "/v1/completions", body)
            wall = time.monotonic() - start
            usage, timings = response["usage"], response.get("timings", {})
            assert usage["prompt_tokens"] == 4096
            assert usage["completion_tokens"] == fixture["max_tokens"]
            tps = timings.get("predicted_per_second") or usage.get(
                "completion_tokens_per_sec"
            )
            assert tps and tps > 0
            arm["samples"].append(
                {
                    "sample": sample + 1,
                    "wall_seconds": wall,
                    "decode_tokens_per_second": tps,
                    "response": response,
                }
            )
            print(f"{fixture['name']} sample {sample + 1}: {tps:.3f} tok/s", flush=True)
        arm["median_decode_tokens_per_second"] = statistics.median(
            s["decode_tokens_per_second"] for s in arm["samples"]
        )
        record["arms"].append(arm)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        # Equal fixed pause between arms, independent of platform telemetry support.
        if fixture is not fixtures["arms"][-1]:
            time.sleep(30)
    record["passed"] = True
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
