#!/usr/bin/env bash
# Request-rate (QPS) sweep with goodput/SLO per model through the SMG gateway,
# using `vllm bench serve` (reused). Produces the load-vs-latency curve + goodput
# that bench/chart.sh renders. No Python from us.
#
# Env:
#   GATEWAY        gateway base url                  [http://localhost:30000]
#   REQUEST_RATES  space-separated req/s to sweep    [8 16 32 64]
#   GOODPUT_SLO    SLO thresholds (ms) for goodput   [ttft:500 tpot:50]
#   NUM_PROMPTS    requests per (model,rate)         [128]
#   MAX_CONCURRENCY in-flight cap (high → rate-bound) [256]
#   IN_LEN/OUT_LEN random token counts               [128/128]
#   MODELS         override; default = discover from gateway
#   OUT_DIR        results dir                        [bench/out]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=scripts/env.sh
. "$SCRIPT_DIR/../scripts/env.sh"

GATEWAY="${GATEWAY:-http://localhost:30000}"
REQUEST_RATES="${REQUEST_RATES:-8 16 32 64}"
GOODPUT_SLO="${GOODPUT_SLO:-ttft:500 tpot:50}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
IN_LEN="${IN_LEN:-128}"
OUT_LEN="${OUT_LEN:-128}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"
mkdir -p "$OUT_DIR"

if [ -n "${MODELS:-}" ]; then
  read -r -a model_arr <<< "$MODELS"
else
  mapfile -t model_arr < <(curl -s "${GATEWAY}/v1/models" | grep -o '"id":"[^"]*"' | sed 's/"id":"//;s/"$//')
fi
[ "${#model_arr[@]}" -gt 0 ] || { echo "[sweep] no models at ${GATEWAY}" >&2; exit 1; }
echo "[sweep] models=${#model_arr[@]} rates='${REQUEST_RATES}' slo='${GOODPUT_SLO}'"

read -r -a rate_arr <<< "$REQUEST_RATES"
read -r -a slo_arr <<< "$GOODPUT_SLO"

for model in "${model_arr[@]}"; do
  for rate in "${rate_arr[@]}"; do
    echo "[sweep] $model @ ${rate} req/s"
    vllm bench serve \
      --backend openai-chat --base-url "$GATEWAY" --endpoint /v1/chat/completions \
      --model "$model" --dataset-name random \
      --num-prompts "$NUM_PROMPTS" --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
      --request-rate "$rate" --max-concurrency "$MAX_CONCURRENCY" \
      --goodput "${slo_arr[@]}" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$OUT_DIR" \
      > "${OUT_DIR}/sweep-$(echo "$model" | tr '/' '_')-r${rate}.log" 2>&1 \
      && echo "[sweep]   ok" || echo "[sweep]   FAILED (see log)"
  done
done
echo "[sweep] done → ${OUT_DIR}; render: bench/chart.sh"
