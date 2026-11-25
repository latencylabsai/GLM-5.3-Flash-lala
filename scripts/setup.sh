#!/usr/bin/env bash
# Prepare the pinned runtime and public checkpoint; leave serving explicit.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)
    backend=mac
    command -v uv >/dev/null || { echo "Install uv before setup." >&2; exit 1; }
    uv sync --locked
    python="$root/.venv/bin/python"
    "$python" -m glm_lala.cli doctor
    bash scripts/setup-mac.sh
    ;;
  Linux/aarch64)
    backend=spark
    bash scripts/setup-spark.sh
    python="$root/.runtime/spark-venv/bin/python"
    "$python" -m glm_lala.cli doctor
    ;;
  *) echo "Supported hosts: Apple Silicon macOS or DGX Spark Linux aarch64." >&2; exit 1 ;;
esac
"$python" -m glm_lala.cli download --output "$root/models/native"
model="$root/models/native"
if [[ "$backend" == mac ]]; then
  for kind in mac mtp; do
    destination="$root/models/$kind"
    if [[ -e "$destination" ]]; then
      # Reuse only a completed pack from this exact pinned native index.
      "$python" - "$root/models/native" "$destination" <<'PY'
import hashlib, json, pathlib, sys
source, pack = map(pathlib.Path, sys.argv[1:])
if not (pack / "COMPLETE").is_file():
    raise SystemExit(f"Incomplete pack at {pack}; move it aside before retrying.")
provenance = json.loads((pack / "PROVENANCE.json").read_text())
digest = hashlib.sha256((source / "model.safetensors.index.json").read_bytes()).hexdigest()
if provenance.get("source_index_sha256") != digest:
    raise SystemExit(f"Pack source differs at {pack}; use a new directory.")
PY
    else
      module=glm_lala.repack
      [[ "$kind" != mtp ]] || module=glm_lala.repack_mtp
      "$python" -m "$module" --source "$root/models/native" --output "$destination"
    fi
  done
  model="$root/models/mac"
fi
mkdir -p "$root/.runtime"
launch=("$python" -m glm_lala.cli serve --backend "$backend" --model "$model")
if [[ "$backend" == mac ]]; then
  runtime_path="${SUSHI_RUNTIME:-$root/.runtime/sushi}"
  [[ "$runtime_path" == /* ]] || runtime_path="$root/$runtime_path"
  launch+=(--binary "$runtime_path/zig-out/bin/sushi")
fi
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\nexec '
  printf '%q ' "${launch[@]}"
  printf '"$@"\n'
} > "$root/.runtime/serve.sh"
echo "Setup complete. Native MTP and vision are enabled. Start serving with:"
printf 'bash %q\n' "$root/.runtime/serve.sh"
