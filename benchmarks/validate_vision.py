"""Check image-conditioned responses with two contrasting synthetic PNGs."""

import argparse
import base64
import hashlib
import json
import pathlib
import struct
import urllib.request
import zlib


def png(rgb):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = (b"\0" + bytes(rgb) * 224) * 224
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", 224, 224, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:43182")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/v1/models") as response:
        model = json.load(response)["data"][0]["id"]
    results = []
    for color, rgb in [("red", (240, 20, 20)), ("blue", (20, 20, 240))]:
        data = png(rgb)
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the dominant color in this image? Answer with just the color name.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(data).decode()
                            },
                        },
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            args.url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            reply = json.load(response)
        content = reply["choices"][0]["message"].get("content") or ""
        results.append(
            {
                "expected": color,
                "image_sha256": hashlib.sha256(data).hexdigest(),
                "response": reply,
                "passed": color in content.lower(),
            }
        )
    record = {
        "model": model,
        "vision_enabled": True,
        "passed": all(r["passed"] for r in results),
        "cases": results,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit("Image color validation failed")


if __name__ == "__main__":
    main()
