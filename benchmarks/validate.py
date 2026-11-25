"""Check a live native server and check paired greedy output agreement."""

import argparse
import datetime
import hashlib
import json
import pathlib
import time
import urllib.request


def request(base, route, payload=None):
    """Read one JSON response and propagate failures."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + route,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def check_api(base, model):
    """Exercise chat, streaming, and recovery after client disconnect."""
    chat = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word hello."}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }
    result = request(base, "/v1/chat/completions", chat)
    assert result["choices"] and result["usage"]["completion_tokens"] > 0
    stream_req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({**chat, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    events, finished = 0, False
    with urllib.request.urlopen(stream_req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = line[6:].strip()
            if event == b"[DONE]":
                finished = True
                break
            chunk = json.loads(event)
            assert "error" not in chunk, chunk
            events += 1
    assert events > 0 and finished, "The stream did not complete."
    # Disconnect after the first event, then verify the next request succeeds.
    with urllib.request.urlopen(stream_req, timeout=180) as response:
        for line in response:
            if line.startswith(b"data: "):
                break
    followup = request(base, "/v1/chat/completions", {**chat, "max_tokens": 8})
    assert followup["choices"]
    return {
        "chat": True,
        "stream": True,
        "stream_events": events,
        "request_after_disconnect": True,
    }


def main():
    """Validate APIs and optionally compare accelerated output to serial."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt-ids", type=pathlib.Path)
    prompts.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--accelerated", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    model = request(base, "/v1/models")["data"][0]["id"]
    api = check_api(base, model)
    prompt = (
        json.loads(args.prompt_ids.read_text())
        if args.prompt_ids
        else args.prompt_file.read_text()
        if args.prompt_file
        else "Write a Python function that checks whether a string is a palindrome."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": args.tokens,
        "seed": 1234,
        "stream": False,
        "ignore_eos": True,
        "enable_mtp": False,
    }
    rows = []
    for accelerated in [False, True] if args.accelerated else [False]:
        started = time.perf_counter()
        completion = request(
            base,
            "/v1/completions",
            {
                **payload,
                "enable_drafter": accelerated,
            },
        )
        text = completion["choices"][0]["text"]
        assert completion["usage"]["completion_tokens"] > 0
        rows.append(
            {
                "accelerated": accelerated,
                "wall_seconds": time.perf_counter() - started,
                "usage": completion["usage"],
                "timings": completion.get("timings"),
                "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "output_token_ids_sha256": completion.get("output_token_ids_sha256"),
                "sample": text[:400],
            }
        )
    same = (
        rows[0]["output_text_sha256"] == rows[1]["output_text_sha256"]
        if len(rows) == 2
        else None
    )
    proof = [row["output_token_ids_sha256"] for row in rows]
    token_match = proof[0] == proof[1] if len(rows) == 2 and all(proof) else None
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "model": model,
        "api": api,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "rows": rows,
        "paired_text_match": same,
        "paired_token_match": token_match,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if same is False or token_match is False:
        raise SystemExit("Accelerated output differs from the serial reference.")


if __name__ == "__main__":
    main()
