# GLM-5.3-Flash-lala

**Runs locally on a 128 GB MacBook Pro or DGX Spark, with native MTP, vision, and 250K context.**

[![Website](https://img.shields.io/badge/Website-Latency_Labs-111827?style=for-the-badge)](https://latencylabs.ai/blog/glm-5.3-flash-lala/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/latencylabs/GLM-5.3-Flash-lala)

Native MTP and the vision encoder are enabled by default on both platforms.
The public Hugging Face checkpoint is pinned to a verified revision.
See [current status](docs/status.md). This package does not contain weights.

The model uses all 288 experts at 2.486 bpw. Mac serving reuses
[Sushi](https://github.com/beamivalice/sushi)'s existing API. Spark serving
reuses ExLlamaV3 and TabbyAPI with the native mixed-K hook.
There is no additional API server or web interface in this repository.

## Setup

Use the same setup command on either supported machine.

| Platform | Validated hardware | Prerequisites |
| --- | --- | --- |
| Mac | M5 Max MacBook Pro, 128 GB RAM | macOS 26.2+, Apple Command Line Tools, Homebrew, Python 3.11+, and `uv` |
| DGX Spark | One NVIDIA GB10, 128 GB unified memory | Python 3.12 with development headers and venv support, a C++ compiler, and CUDA 13.0 |

On DGX Spark, the system image may not include the Python 3.12 development
headers. If setup reports that they are missing, install them:

~~~bash
sudo apt install python3.12-dev python3.12-venv
~~~

Both platforms need git and space for the approximately 100 GB checkpoint.
Mac also needs approximately 90 GB for the main repack and 3.8 GB for its MTP
pack. Other Mac configurations are not yet qualified. Full Xcode is not
required; MLX source builds are not supported.

The model is public. No Hugging Face login is required to download it.

### 1. Install

~~~bash
git clone https://github.com/latencylabsai/GLM-5.3-Flash-lala.git
cd GLM-5.3-Flash-lala
bash scripts/setup.sh
~~~

The installer detects your platform, prepares the pinned runtime, downloads
and verifies the checkpoint, and writes the launch command. On Mac it downloads
verified libraries from the pinned Sushi release, compiles patched Sushi, and
prepares the main and native-MTP packs. On Spark it builds pinned ExLlamaV3
and installs TabbyAPI. No separate download, repack, or MTP setup is needed.

### 2. Start the server

~~~bash
bash .runtime/serve.sh
~~~

The server listens at **http://127.0.0.1:8000** with native MTP and vision
enabled, one active stream, **262,144-token context capacity**, and 16-bit KV
storage. Native MTP uses four draft tokens; suffix drafting is off. No DFlash2
model is required. Add `--serial` to this launch command for reference decoding.

### 3. Ask a question

In another terminal, detect the model ID and send a question. This works on
both Mac and DGX Spark:

~~~bash
MODEL=$(curl -fsS http://127.0.0.1:8000/v1/models | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Why is the sky blue? Answer briefly.\"}],
    \"max_tokens\": 1024,
    \"stream\": false
  }"
~~~

The answer is in `choices[0].message.content` in the JSON response. Replace
the question in `content` with your own. Both `/v1/chat/completions` and
`/v1/completions` support streaming. For images, send OpenAI-style `image_url`
content (including PNG data URLs) to `/v1/chat/completions`.

For upgrades or custom runtime locations, see the [Mac guide](docs/mac.md)
or [Spark guide](docs/spark.md). For smoke tests and benchmark reproduction,
see [validation](docs/validation.md). These are advanced details; the setup
above handles both platforms.

## Token speed

Median decode speed with 4K-token prompts, native MTP, and vision enabled;
prompt processing is excluded.

| Hardware | Code (tok/s) | Prose (tok/s) |
| --- | ---: | ---: |
| MacBook Pro, M5 Max, 128 GB | **52.23** | **28.88** |
| DGX Spark, 128 GB | **41.97** | **19.91** |

[Benchmark conditions and results](docs/benchmarks.md).

## Long-context support

Both platforms have passed **at least 250,000 input tokens**, generation, and
a follow-up request with native MTP and vision loaded. The configured
262,144-token capacity includes both input and output.

These checks use repeated public text to validate capacity and execution;
they do not establish long-context retrieval quality. See
[validation commands](docs/validation.md#long-context-capacity) and
[installation and validation results](docs/status.md).

## Acknowledgements

This package builds on the work of the following teams and communities:

- **Z.ai and the GLM team** for the original
  [GLM-5.3-Flash model](https://huggingface.co/zai-org/GLM-5.3-Flash).
  GLM-5.3-Flash-lala is derived from their model; credit for the original
  architecture, training, and model release belongs to its authors.
- **turboderp and the ExLlamaV3 contributors** for
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3), the EXL3
  quantization format and inference engine used on DGX Spark.
- **The TabbyAPI contributors** for
  [TabbyAPI](https://github.com/theroyallab/tabbyAPI), which provides the
  existing HTTP serving API on Spark.
- **The Sushi contributors and David Dalcu's mlx-serve project** for
  [Sushi](https://github.com/beamivalice/sushi) and its upstream
  [mlx-serve](https://github.com/ddalcu/mlx-serve), which underpin the native
  Mac runtime and serving API.
- **Apple's MLX team and contributors** for
  [MLX](https://github.com/ml-explore/mlx) and
  [mlx-c](https://github.com/ml-explore/mlx-c), used by the Mac runtime.
- **The PyTorch and Triton contributors** for
  [PyTorch](https://github.com/pytorch/pytorch) and
  [Triton](https://github.com/triton-lang/triton), used by the Spark runtime,
  and **Hugging Face** for model hosting and
  [huggingface_hub](https://github.com/huggingface/huggingface_hub), used to
  download pinned checkpoint revisions.

See the retained [Sushi notices](patches/SUSHI-NOTICE) for additional
upstream credits and component licenses, and the retained
[ExLlamaV3 license](patches/EXLLAMAV3-LICENSE). Model weights retain their publishers' licenses; this package's license
does not replace them.

## Reproducibility and licenses

[runtime-lock.json](runtime-lock.json) identifies source revisions. The Mac
patch is consolidated against its exact Sushi base. The package does not
depend on the internal local-lab repository at runtime.

Sushi's MIT license and notices are retained under `patches/`. Its transitive
dependencies and model artifacts retain their respective licenses.
No model weights, credentials, or private machine addresses are included.
The package's own code is MIT-licensed; see [LICENSE](LICENSE).
