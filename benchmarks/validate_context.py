"""Exercise a real long prompt through the packaged server under a host GPU lock."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def request(url, body=None, timeout=10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(2000).decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mac", "spark"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--mtp-pack", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--prompt-tokens", type=int, default=250000)
    parser.add_argument("--port", type=int, default=43180)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--check-api-and-vision", action="store_true")
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.prompt_tokens + 32 > args.context:
        parser.error("Prompt plus output must fit in the configured context.")
    args.output.mkdir(parents=True, exist_ok=False)
    # Public fixture only. Remove chat/control IDs before repeating the text tokens.
    fixture = json.loads((ROOT / "benchmarks/public-copy.ids.json").read_text())
    body = [token for token in fixture if token < 154000]
    prompt = (body * ((args.prompt_tokens // len(body)) + 1))[: args.prompt_tokens]
    command = [
        sys.executable,
        "-m",
        "glm_lala.cli",
        "serve",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--context",
        str(args.context),
        "--port",
        str(args.port),
    ]
    if args.serial:
        command += ["--serial"]
    if args.mtp_pack:
        if args.backend != "mac":
            parser.error("--mtp-pack is a Mac option")
        command += ["--mtp-pack", str(args.mtp_pack)]
    if args.binary:
        command += ["--binary", str(args.binary)]
    record = {
        "native_mtp": not args.serial,
        "backend": args.backend,
        "context_capacity": args.context,
        "requested_prompt_tokens": len(prompt),
        "kv_precision": "16-bit",
        "fixture": "public-copy.ids.json, control IDs removed, repeated and truncated",
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "passed": False,
        "runtime_lock": json.loads((ROOT / "runtime-lock.json").read_text()),
    }
    base = f"http://127.0.0.1:{args.port}"
    stopped = threading.Event()

    def progress():
        while not stopped.wait(30):
            if args.backend == "spark":
                print("Long-context request still running", flush=True)
                continue
            try:
                gauges = request(base + "/metrics.json", timeout=5)["gauges"]
                sample = {
                    key: gauges[key]
                    for key in (
                        "prefill_tokens_live",
                        "prefill_tokens_expected",
                        "memory_mb",
                        "mlx_active_bytes",
                    )
                    if key in gauges
                }
                record.setdefault("memory_samples", []).append(sample)
                print(f"Long-context progress: {sample}", flush=True)
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                ValueError,
                RuntimeError,
            ):
                print("Long-context request still running", flush=True)

    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited with {server.returncode}; see server.log"
                    )
                try:
                    models = request(base + "/v1/models")
                    break
                except (urllib.error.URLError, TimeoutError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup exceeded 600 seconds")
                    time.sleep(2)
            model = models["data"][0]["id"]
            if args.check_api_and_vision:
                from validate import check_api

                record["api"] = check_api(base, model)
                vision_output = args.output / "vision.json"
                subprocess.run(
                    [sys.executable, str(ROOT / "benchmarks/validate_vision.py"),
                     "--url", base, "--output", str(vision_output)],
                    check=True, capture_output=True, text=True,
                )
                record["vision"] = json.loads(vision_output.read_text())
                print("Chat, streaming, disconnect recovery, and red/blue vision checks passed.", flush=True)
            if args.backend == "spark":
                # TabbyAPI's completion endpoint accepts text, not token arrays.
                # Round-trip through its own tokenizer and check the actual size.
                text = request(
                    base + "/v1/token/decode", {"tokens": prompt}, timeout=60
                )["text"]
                while True:
                    encoded = request(
                        base + "/v1/token/encode", {"text": text}, timeout=60
                    )
                    if encoded["length"] >= args.prompt_tokens:
                        break
                    extra = body * (
                        (args.prompt_tokens - encoded["length"]) // len(body) + 1
                    )
                    text += request(
                        base + "/v1/token/decode", {"tokens": extra}, timeout=60
                    )["text"]
                if encoded["length"] + 32 > args.context:
                    raise ValueError("Round-tripped text exceeds context capacity")
                record["submitted_prompt_tokens"] = encoded["length"]
                record["prompt_sha256"] = hashlib.sha256(
                    json.dumps(encoded["tokens"]).encode()
                ).hexdigest()
                record["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                record["input_format"] = "text, counted by TabbyAPI's tokenizer"
                prompt = text
            else:
                record["submitted_prompt_tokens"] = len(prompt)
                record["input_format"] = "token IDs"
            print(
                f"Server ready; submitting {record['submitted_prompt_tokens']} input tokens",
                flush=True,
            )
            monitor = threading.Thread(target=progress, daemon=True)
            monitor.start()
            started = time.monotonic()
            result = request(
                base + "/v1/completions",
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                    "stream": False,
                },
                timeout=14400,
            )
            record.update(elapsed_seconds=time.monotonic() - started, response=result)
            usage = result.get("usage", {})
            if usage.get("prompt_tokens") != record["submitted_prompt_tokens"]:
                raise RuntimeError(f"Prompt was truncated or unreported: {usage}")
            if usage.get("completion_tokens", 0) < 1:
                raise RuntimeError("No output token was generated")
            followup = request(
                base + "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
                timeout=300,
            )
            record.update(followup_usage=followup.get("usage"), passed=True)
        except Exception as error:
            record["error"] = str(error)
            raise
        finally:
            stopped.set()
            if "monitor" in locals():
                monitor.join(timeout=6)
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            (args.output / "result.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
