#!/usr/bin/env bash
# Benchmark the SMG gateway using `vllm bench serve` (reused; no Python from us).
# Discovers the models behind the gateway and loads each — sequentially
# (per-model) or all at once (concurrent = multi-tenant interference).
#
# Env:
#   GATEWAY          gateway base url                 [http://localhost:30000]
#   MODE             per-model | concurrent           [per-model]
#   NUM_PROMPTS      requests per model               [64]
#   MAX_CONCURRENCY  in-flight cap per model          [8]
#   REQUEST_RATE     req/s (inf = as fast as possible)[inf]
#   IN_LEN/OUT_LEN   random prompt/gen token counts   [128/128]
#   OUT_DIR          where results land               [bench/out]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=scripts/env.sh
. "$SCRIPT_DIR/../scripts/env.sh"   # venv (vllm) on PATH

GATEWAY="${GATEWAY:-http://localhost:30000}"
MODE="${MODE:-per-model}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
IN_LEN="${IN_LEN:-128}"
OUT_LEN="${OUT_LEN:-128}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"
mkdir -p "$OUT_DIR"

# Discover models behind the gateway.
mapfile -t MODELS < <(curl -s "${GATEWAY}/v1/models" | grep -o '"id":"[^"]*"' | sed 's/"id":"//;s/"$//')
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "[bench] no models at ${GATEWAY}/v1/models — is the gateway up?" >&2
  exit 1
fi
echo "[bench] gateway=$GATEWAY mode=$MODE models=${#MODELS[@]}: ${MODELS[*]}"

bench_one() {
  local model="$1" tag="$2"
  echo "[bench] -> $model (tag=$tag)"
  vllm bench serve \
    --backend openai-chat \
    --base-url "$GATEWAY" \
    --endpoint /v1/chat/completions \
    --model "$model" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$IN_LEN" \
    --random-output-len "$OUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT_DIR" \
    > "${OUT_DIR}/${tag}.log" 2>&1
  echo "[bench]    done -> ${OUT_DIR}/${tag}.log"
}

i=0
if [ "$MODE" = "concurrent" ]; then
  echo "[bench] CONCURRENT: all models loaded simultaneously (interference test)"
  for m in "${MODELS[@]}"; do bench_one "$m" "concurrent-$i" & i=$((i + 1)); done
  wait
else
  for m in "${MODELS[@]}"; do bench_one "$m" "permodel-$i"; i=$((i + 1)); done
fi

echo "[bench] === summary (throughput + latency) ==="
grep -hE "Successful requests|Request throughput|Output token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT" "${OUT_DIR}"/*.log 2>/dev/null || true
echo "[bench] full results + JSON in ${OUT_DIR}"
