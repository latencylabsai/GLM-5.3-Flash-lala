# Validation

Install and launch using the [README setup](../README.md). Run these commands
from the repository root. Use the installed Python environment for your platform:

~~~bash
# Mac
source .venv/bin/activate
# Spark
source .runtime/spark-venv/bin/activate
~~~

Run only the activation command for your machine. The running-server checks
below use port 8000. Stop that server before a capacity test, which launches
its own instance. Reserve the GPU for benchmarking.

## Running-server checks

```bash
python benchmarks/validate.py --output benchmarks/records/local-smoke.json
# Mac native-MTP paired check:
python benchmarks/validate.py --accelerated \
  --output benchmarks/records/local-paired.json
```

Validation checks chat, streamed chat, a request after a client disconnect,
and raw completions. On Mac, a paired run records serial/accelerated output hashes
and fails if the outputs differ. Backend decode time is recorded separately
from total HTTP request time. Use the [4K benchmark results](benchmarks.md) for token-speed measurements.

The `--accelerated` flag controls Sushi per request. For paired Spark token
equivalence, use the runner in the [Spark guide](spark.md#validate), which
starts separate serial and native-MTP servers.

## Long-context capacity

Both launchers default to **262,144 tokens of context capacity**, including
input and generated tokens, with one active stream and 16-bit cache storage.
The long-context acceptance check submits at least 250,000 input tokens,
checks that usage reports the full input, generates output, and sends a
follow-up request. It uses token IDs on Mac and tokenizer-counted text on
Spark. It launches and stops its own server:

```bash
python benchmarks/validate_context.py --backend mac --model models/mac \
  --output /tmp/lala-mac-250k
python benchmarks/validate_context.py --backend spark --model models/native \
  --output /tmp/lala-spark-250k
```

Both commands exercise default native MTP with vision loaded. Both hosts
have passed at least 250K input tokens plus generation and a follow-up in
this configuration. Add `--serial` to check the reference path.
Fresh public-download installations passed these checks on both platforms;
see the [installation records](status.md).

Run only the command for your machine, with no other model occupying its GPU.
The fixture repeats public text tokens to test capacity and execution; it is
not a long-context quality benchmark. See [current status](status.md) for
installation and validation results.

## Multimodal and matched 4K checks

Both launchers enable the checkpoint's vision tower. Send OpenAI-style
`image_url` content (including a PNG data URL) to `/v1/chat/completions`.
Check image-conditioned responses with:

```bash
python benchmarks/validate_vision.py --url http://127.0.0.1:8000 --output vision.json
```

The matched code/prose fixtures come from Sushi's native release benchmark:
4,096 prompt tokens, three greedy samples, 256 generated code tokens and
128 prose tokens. See [benchmark results and protocol](benchmarks.md).

