# DGX Spark

Follow the [README setup](../README.md#setup) to install and start serving.

Validated on **one NVIDIA GB10 with 128 GB unified memory**. Requires Python
3.12 with venv support and development headers, a C++ compiler, and CUDA 13.0.
On Ubuntu, install the headers and venv support with
`sudo apt install python3.12-dev python3.12-venv`.

Setup installs patched ExLlamaV3 and TabbyAPI into `.runtime/spark-venv`.
It uses the native EXL3 checkpoint directly, without repacking. Allow
approximately 100 GB for the checkpoint, plus runtime and build dependencies.

Native MTP and vision are enabled by default, with one active stream and
262,144-token context capacity. The local API uses model ID `lala` at
`http://127.0.0.1:8000`. Configure TabbyAPI authentication before sharing it.

## Validate

With no other model server using the GPU, run:

```bash
bash scripts/compare-spark-4k.sh models/native /tmp/lala-spark-paired
```

This checks the code/prose outputs and records token speed. See
[validation](validation.md) for API, vision, and capacity checks, and
[token speed](benchmarks.md) for results. Runtime versions are pinned in
[runtime-lock.json](../runtime-lock.json).
