#!/usr/bin/env bash
# Routing A/B: SMG cache_aware vs round_robin with 2 replicas of one model under
# CACHE PRESSURE. Each replica's KV cache is constrained (low --gpu-memory-
# utilization + small --max-model-len) and the workload uses many distinct shared
# prefixes, so the aggregate prefix working set exceeds a single replica's cache.
# Expectation: cache_aware pins each prefix to one replica (each caches ~half →
# fits → hits) while round_robin makes both replicas cache all prefixes (eviction
# thrash) → cache_aware wins on throughput / TTFT.
#
# Cold cache per policy: replicas are torn down (by PID — no pkill, avoids the
# pattern self-match footgun) and relaunched between runs.
#
# Requires ports 8001/8002/30000 free at start (run `scripts/stop.sh` first).
# Env: MODEL, GMU [0.15], MAXLEN [2560], PREFIX_LEN [2048], OUT_LEN [16],
#      NUM_PREFIXES [200], NUM_PROMPTS [800], CONC [16], OUT_DIR.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/env.sh
. "$REPO/scripts/env.sh"

# NOTE: GMU must be high enough for the worker to init KV at all (~0.25 min for a
# 0.5B on 16 GB after weights + CUDA context), yet low enough that NUM_PREFIXES
# prefixes overflow one replica's cache. 0.30 + 256 prefixes is a working point.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GMU="${GMU:-0.30}"
MAXLEN="${MAXLEN:-2560}"
PREFIX_LEN="${PREFIX_LEN:-2048}"
OUT_LEN="${OUT_LEN:-16}"
NUM_PREFIXES="${NUM_PREFIXES:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-768}"
CONC="${CONC:-16}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/out}"; mkdir -p "$OUT_DIR"
RUN_DIR="$REPO/run"; mkdir -p "$RUN_DIR"
PIDF="$RUN_DIR/routing.pids"
PORTS=(8001 8002)

launch_replicas() {
  : > "$PIDF"
  local port
  for port in "${PORTS[@]}"; do
    setsid env CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" --port "$port" \
      --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" --enforce-eager \
      --trust-remote-code > "$RUN_DIR/rep-$port.log" 2>&1 < /dev/null &
    echo "$!" >> "$PIDF"
  done
  for port in "${PORTS[@]}"; do
    for _ in $(seq 1 180); do
      [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" 2>/dev/null || echo 000)" = "200" ] && break
      sleep 2
    done
  done
}

kill_replicas() {
  [ -f "$PIDF" ] || return 0
  while read -r p; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done < "$PIDF"
  sleep 4
}

run_policy() {
  local pol="$1"
  launch_replicas
  smg launch --worker-urls "http://localhost:${PORTS[0]}" "http://localhost:${PORTS[1]}" \
    --policy "$pol" --port 30000 > "$OUT_DIR/gw-$pol.log" 2>&1 &
  local gw=$!
  for _ in $(seq 1 30); do curl -s -m2 http://localhost:30000/v1/models 2>/dev/null | grep -q . && break; sleep 1; done
  echo "###### policy=$pol  (gmu=$GMU maxlen=$MAXLEN prefixes=$NUM_PREFIXES prefix_len=$PREFIX_LEN) ######"
  vllm bench serve --backend openai-chat --base-url http://localhost:30000 --endpoint /v1/chat/completions \
    --model "$MODEL" --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len "$PREFIX_LEN" --prefix-repetition-suffix-len 32 \
    --prefix-repetition-output-len "$OUT_LEN" --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" 2>&1 \
    | grep -E "Output token throughput|Request throughput|Mean TTFT|P99 TTFT"
  kill "$gw" 2>/dev/null || true
  kill_replicas
}

echo "[routing_ab] model=$MODEL  pressure: gmu=$GMU maxlen=$MAXLEN prefixes=$NUM_PREFIXES"
run_policy cache_aware
run_policy round_robin
echo "[routing_ab] done — logs in $OUT_DIR/gw-*.log"
