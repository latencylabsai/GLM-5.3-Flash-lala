#!/usr/bin/env bash
# Build an isolated Sushi checkout; never modify an existing runtime.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
  echo "Apple Silicon macOS is required." >&2; exit 1;
}
if [[ -n "${SUSHI_MLX_STAGE:-}" ]]; then
  echo "SUSHI_MLX_STAGE is no longer supported; setup uses verified Sushi release libraries." >&2
  exit 1
fi
xcrun --sdk macosx --show-sdk-path >/dev/null || {
  echo "Install Apple Command Line Tools with xcode-select --install." >&2; exit 1;
}
command -v brew >/dev/null || { echo "Install Homebrew first." >&2; exit 1; }
python="$root/.venv/bin/python"
[[ -x "$python" ]] || python=python3
"$python" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required; run uv sync --locked first"'
source_repo="${SUSHI_SOURCE:-https://github.com/beamivalice/sushi.git}"
runtime="${SUSHI_RUNTIME:-$root/.runtime/sushi}"
[[ "$runtime" == /* ]] || runtime="$root/$runtime"
revision=fc2fd843c2931f7cac150cdc86bb3873de273458
mkdir -p "$root/.runtime"
if [[ ! -d "$runtime" ]]; then
  git clone --no-hardlinks "$source_repo" "$runtime"
  git -C "$runtime" checkout --detach "$revision"
fi
[[ "$(git -C "$runtime" rev-parse HEAD)" == "$revision" ]] || {
  echo "Unexpected Sushi revision. Use a new runtime directory." >&2; exit 1;
}
if git -C "$runtime" apply --check "$root/patches/sushi-lala.patch" 2>/dev/null; then
  git -C "$runtime" apply "$root/patches/sushi-lala.patch"
elif git -C "$runtime" apply --reverse --check "$root/patches/sushi-lala.patch" 2>/dev/null; then
  : # Already matches the current patch.
else
  echo "This checkout has a different patch. Set SUSHI_RUNTIME to a new directory to preserve the existing runtime." >&2
  exit 1
fi
cd "$runtime"
if [[ -n "${SUSHI_ZIG_STAGE:-}" ]]; then
  if [[ ! -e .zig-toolchain ]]; then
    ln -s "$SUSHI_ZIG_STAGE" .zig-toolchain
  fi
  [[ "$(.zig-toolchain/zig version)" == 0.17.1 ]]
else
  bash scripts/fetch-zig.sh
fi
brew install webp
"$python" "$root/scripts/stage-sushi-libraries.py" "$runtime"
.zig-toolchain/zig build -Doptimize=ReleaseFast -Darchs=glm5
zig-out/bin/sushi --version
shasum -a 256 zig-out/bin/sushi "$root/patches/sushi-lala.patch"
