#!/usr/bin/env bash
# Install into a dedicated environment; never change the system Python.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
  echo "This installer targets one DGX Spark (Linux aarch64)." >&2; exit 1;
}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
command -v nvcc >/dev/null || { echo "Install the CUDA 13.0 toolkit first." >&2; exit 1; }
nvcc --version | grep -q 'release 13.0,' || {
  echo "The pinned Spark build requires CUDA toolkit 13.0." >&2; exit 1;
}
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the pinned Spark environment"'
python_include=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
if ! printf '#include <Python.h>\n' | c++ -I"$python_include" -E -x c++ - >/dev/null 2>&1; then
  echo "Install Python 3.12 development headers (Ubuntu: python3.12-dev) first." >&2
  echo "For locally extracted headers, set CPATH to their include directories." >&2
  exit 1
fi
export TORCH_CUDA_ARCH_LIST=12.1a
export MAX_JOBS=${MAX_JOBS:-4}
runtime="$root/.runtime"
mkdir -p "$runtime"
checkout() {
  local url=$1 dest=$2 revision=$3 source_patch=${4:-}
  if [[ ! -d "$dest" ]]; then
    git clone "$url" "$dest"
    git -C "$dest" checkout --detach "$revision"
  fi
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$revision" ]] || {
    echo "Unexpected source revision in $dest." >&2; exit 1;
  }
  if [[ -n "$(git -C "$dest" status --porcelain --untracked-files=no)" ]]; then
    # A repeated installation may contain exactly our pinned patch, but must
    # not overwrite or silently build unrelated local source changes.
    if [[ -z "$source_patch" || "$(git -C "$dest" diff --binary --full-index --no-color --no-ext-diff --no-textconv --src-prefix=a/ --dst-prefix=b/ --unified=3 HEAD)" != "$(cat "$source_patch")" ]]; then
      echo "Unexpected tracked source changes in $dest; use a clean checkout." >&2; exit 1
    fi
  elif [[ -n "$source_patch" ]]; then
    git -C "$dest" apply --check "$source_patch"
    git -C "$dest" apply "$source_patch"
  fi
}
checkout https://github.com/turboderp-org/exllamav3.git "$runtime/exllamav3" 058b8630fd081f3ff46fce6246955b157c3a2e3b "$root/patches/exllamav3-consistent-decode.patch"
checkout https://github.com/theroyallab/tabbyAPI.git "$runtime/tabby-native" cdea8acc4ccabe795863772e690d1357298cef74
python3 -m venv "$runtime/spark-venv"
export PATH="$runtime/spark-venv/bin:$PATH"
python="$runtime/spark-venv/bin/python"
"$python" -m pip install 'pip==26.3' 'setuptools==80.10.2' 'wheel==0.48.0' 'ninja==1.13.2'
"$python" -m pip install -c "$root/configs/spark-native-constraints.txt" 'torch==2.13.0+cu130' --index-url https://download.pytorch.org/whl/cu130
"$python" -m pip install --no-build-isolation -c "$root/configs/spark-native-constraints.txt" 'triton==3.7.1' 'uvloop==0.22.1' "$runtime/tabby-native" -e "$root"
"$python" -m pip install --no-build-isolation -c "$root/configs/spark-native-constraints.txt" "$runtime/exllamav3"
"$python" -c 'import torch, exllamav3_ext; assert getattr(exllamav3_ext, "lala_consistent_decode_version", None) == 1, "The installed ExLlamaV3 extension is missing the Lala decode correction"'
if ! dependency_check=$("$python" -m pip check 2>&1); then
  # NVIDIA's aarch64 wheel has an internal SBSA platform tag. pip accepts
  # its filename at install, but flags that internal tag during pip check.
  # Allow only this exact metadata warning; all dependency conflicts fail.
  if [[ "$dependency_check" != 'nvidia-cusparselt-cu13 0.8.1 is not supported on this platform' ]]; then
    echo "$dependency_check" >&2; exit 1
  fi
  echo "Known NVIDIA cuSPARSELt 0.8.1 SBSA wheel-tag warning; checking GPU execution."
else
  echo "$dependency_check"
fi
"$python" -c 'import torch; from exllamav3.version import __version__; print("ExLlamaV3", __version__, "PyTorch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
"$python" -c 'import torch; x=torch.ones((32,32), device="cuda"); assert torch.all(x @ x == 32).item(); torch.cuda.synchronize(); print("CUDA matrix multiply passed")'
"$python" -m pip freeze > "$runtime/spark-installed.txt"
LALA_RUNTIME_DIR="$runtime" "$python" -c 'import json, os, pathlib; pathlib.Path(os.environ["LALA_RUNTIME_DIR"], "spark-build-env.json").write_text(json.dumps({key: os.environ[key] for key in ("CPATH", "CUDA_HOME") if os.environ.get(key)}, indent=2) + "\n")'
echo "Ready: $python -m glm_lala.cli serve --backend spark --model /path/to/native/checkpoint"
