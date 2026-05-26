#!/usr/bin/env bash
# Prefix-cache A/B against the gateway: same model, RANDOM (no shared prefix) vs
# PREFIX_REPETITION (shared prefix). Quantifies vLLM automatic prefix caching
# (APC), which skips re-prefilling shared prefixes. The win scales with PREFIX_LEN
# and how prefill-dominated the workload is (small OUTPUT_LEN), e.g. ~2x throughput
# and ~-64% P99 TTFT with a 2048-token shared prefix on the RTX 5080.
#
# NOTE: this measures APC *inside one worker*. SMG's cache-aware *routing* is a
# separate lever that only matters with MULTIPLE replicas of a model (route
# same-prefix requests to the same replica) — see docs/benchmarking.md.
#
# Env: GATEWAY, MODEL (default: first discovered), PREFIX_LEN [2048], SUFFIX_LEN
#      [32], OUTPUT_LEN [32], NUM_PREFIXES [2], NUM_PROMPTS [128], MAX_CONCURRENCY
#      [16], OUT_DIR.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=scripts/env.sh
. "$SCRIPT_DIR/../scripts/env.sh"

GATEWAY="${GATEWAY:-http://localhost:30000}"
PREFIX_LEN="${PREFIX_LEN:-2048}"
SUFFIX_LEN="${SUFFIX_LEN:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
NUM_PREFIXES="${NUM_PREFIXES:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-16}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"
mkdir -p "$OUT_DIR"
IN_LEN=$((PREFIX_LEN + SUFFIX_LEN))

if [ -z "${MODEL:-}" ]; then
  MODEL="$(curl -s "${GATEWAY}/v1/models" | grep -o '"id":"[^"]*"' | head -1 | sed 's/"id":"//;s/"$//')"
fi
[ -n "$MODEL" ] || { echo "[prefix_ab] no model at ${GATEWAY}" >&2; exit 1; }
echo "[prefix_ab] model=$MODEL prefix=$PREFIX_LEN suffix=$SUFFIX_LEN out=$OUTPUT_LEN prefixes=$NUM_PREFIXES"

_metrics() { grep -E "Request throughput|Output token throughput|Mean TTFT|P99 TTFT" "$1"; }

echo "=== A) RANDOM (${IN_LEN}-in / ${OUTPUT_LEN}-out, no shared prefix) ==="
vllm bench serve --backend openai-chat --base-url "$GATEWAY" --endpoint /v1/chat/completions \
  --model "$MODEL" --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY" \
  > "${OUT_DIR}/prefixab-random.log" 2>&1
_metrics "${OUT_DIR}/prefixab-random.log"

echo "=== B) PREFIX_REPETITION (${PREFIX_LEN}-prefix x${NUM_PREFIXES} / ${OUTPUT_LEN}-out, APC hits) ==="
vllm bench serve --backend openai-chat --base-url "$GATEWAY" --endpoint /v1/chat/completions \
  --model "$MODEL" --dataset-name prefix_repetition --prefix-repetition-prefix-len "$PREFIX_LEN" \
  --prefix-repetition-suffix-len "$SUFFIX_LEN" --prefix-repetition-output-len "$OUTPUT_LEN" \
  --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY" \
  > "${OUT_DIR}/prefixab-prefix.log" 2>&1
_metrics "${OUT_DIR}/prefixab-prefix.log"

echo "[prefix_ab] logs in ${OUT_DIR}/prefixab-*.log"
