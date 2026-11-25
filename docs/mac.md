# MacBook Pro

Follow the [README setup](../README.md#setup) to install and start serving.

Validated on an **M5 Max MacBook Pro with 128 GB RAM**. Requires macOS 26.2+,
Apple Command Line Tools, Homebrew, Python 3.11+, and `uv`.

Setup downloads verified Sushi release libraries, builds patched Sushi, and
prepares the model and native-MTP packs. **Full Xcode and MLX source builds
are not required.** Repacking preserves the checkpoint's quantized weights.
Allow approximately 200 GB for the checkpoint and model packs, plus runtime
and build dependencies.

Native MTP and vision are enabled by default, with one active stream and
262,144-token context capacity. Pass `--port` or `--context` to
`bash .runtime/serve.sh` to change the defaults.

See [validation](validation.md) for checks and [token speed](benchmarks.md)
for benchmark results. Runtime versions are pinned in
[runtime-lock.json](../runtime-lock.json).
