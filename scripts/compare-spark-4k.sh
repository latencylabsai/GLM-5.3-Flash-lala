#!/usr/bin/env bash
# Run under an exclusive GPU lease. The script owns only the servers it starts.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
model=$(realpath "${1:?Usage: compare-spark-4k.sh MODEL OUTPUT_DIR [PORT]}")
out=${2:?Supply a new output directory}
[[ ! -e "$out" ]] || { echo "Output directory already exists: $out" >&2; exit 1; }
mkdir -p "$out"
out=$(realpath "$2")
port=${3:-43184}
python="$root/.runtime/spark-venv/bin/python"
server=''
cleanup() {
  if [[ -n "$server" ]]; then
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
    server=''
  fi
}
trap cleanup EXIT
for mode in serial mtp; do
  LALA_TRACE="$out/$mode-tokens.jsonl" "$python" benchmarks/spark_token_server.py \
    serve --backend spark --model "$model" --spark-draft "$mode" --port "$port" \
    > "$out/$mode-server.log" 2>&1 &
  server=$!
  ready=false
  for ((attempt=0; attempt<300; attempt++)); do
    kill -0 "$server" || { tail -30 "$out/$mode-server.log"; exit 1; }
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 2
  done
  "$ready" || { echo "Server did not become ready." >&2; exit 1; }
  if [[ "$mode" == serial ]]; then bench_mode=serial; else bench_mode=accelerated; fi
  "$python" benchmarks/compare_4k.py --url "http://127.0.0.1:$port" \
    --backend spark --mode "$bench_mode" --fixtures benchmarks/fixtures/sushi-4k.json \
    --output "$out/$mode-bench.json"
  cleanup
  sleep 5
done
"$python" benchmarks/check_spark_outputs.py "$out"
