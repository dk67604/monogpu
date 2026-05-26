#!/usr/bin/env bash
# Controlled co-location experiment: isolate the two confounds (GPU-memory split
# and contention) while holding benchmark concurrency and CUDA graphs CONSTANT.
# CUDA graphs are ON (no --enforce-eager) and total in-flight requests are equal
# across configs, so the comparison is apples-to-apples. Benches hit the worker
# DIRECTLY (no gateway) to measure model throughput, not routing overhead.
#
#   A) 0.5B DEDICATED  : 1 worker, gmu 0.85, conc 48          (full GPU)
#   B) 0.5B MEM-SPLIT  : 1 worker, gmu 0.30, conc 48          (1/3 memory, no neighbors)
#   C) 3 models SHARED : gmu 0.30 each, conc 16 each (=48 tot), benched concurrently
#
# A vs B = effect of the memory/load setting alone. B vs C = effect of contention.
# PID-based teardown (no pkill — avoids the pattern self-match footgun).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/env.sh
. "$REPO/scripts/env.sh"

NUM_PROMPTS="${NUM_PROMPTS:-512}"
IN_LEN="${IN_LEN:-128}"; OUT_LEN="${OUT_LEN:-128}"
MAXLEN="${MAXLEN:-4096}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out}"; mkdir -p "$OUT_DIR"
RUN_DIR="$REPO/run"; mkdir -p "$RUN_DIR"
PIDF="$RUN_DIR/coloc.pids"
SMALL=Qwen/Qwen2.5-0.5B-Instruct

start_worker() { # model port gmu
  setsid env CUDA_VISIBLE_DEVICES=0 vllm serve "$1" --port "$2" \
    --gpu-memory-utilization "$3" --max-model-len "$MAXLEN" --trust-remote-code \
    > "$RUN_DIR/coloc-$2.log" 2>&1 < /dev/null &
  echo "$!" >> "$PIDF"
}
wait_health() { # port
  for _ in $(seq 1 240); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$1/health" 2>/dev/null || echo 000)" = "200" ] && return 0
    sleep 2
  done; echo "[coloc] WARN port $1 not healthy"; }
kill_all() { [ -f "$PIDF" ] && while read -r p; do kill "$p" 2>/dev/null || true; done < "$PIDF"; sleep 5; : > "$PIDF"; }
bench() { # port model conc tag
  vllm bench serve --backend openai-chat --base-url "http://localhost:$1" \
    --endpoint /v1/chat/completions --model "$2" --dataset-name random \
    --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$3" --request-rate inf \
    > "$OUT_DIR/coloc-$4.log" 2>&1
  echo -n "  [$4] "; grep -E "Output token throughput" "$OUT_DIR/coloc-$4.log" | tail -1
}

: > "$PIDF"
echo "=== A) 0.5B DEDICATED (gmu 0.85, conc 48, graphs ON) ==="
start_worker "$SMALL" 8001 0.85; wait_health 8001; bench 8001 "$SMALL" 48 "A-dedicated"; kill_all

echo "=== B) 0.5B MEM-SPLIT (gmu 0.30, conc 48, graphs ON, no neighbors) ==="
start_worker "$SMALL" 8001 0.30; wait_health 8001; bench 8001 "$SMALL" 48 "B-memsplit"; kill_all

echo "=== C) 3 MODELS SHARED (gmu 0.30 each, conc 16 each = 48 total, concurrent) ==="
start_worker "$SMALL" 8001 0.30; wait_health 8001
start_worker Qwen/Qwen2.5-1.5B-Instruct 8002 0.30; wait_health 8002
start_worker HuggingFaceTB/SmolLM2-360M-Instruct 8003 0.30; wait_health 8003
bench 8001 "$SMALL" 16 "C-0.5B" &
bench 8002 Qwen/Qwen2.5-1.5B-Instruct 16 "C-1.5B" &
bench 8003 HuggingFaceTB/SmolLM2-360M-Instruct 16 "C-360M" &
wait
echo "  C aggregate tok/s:"; grep -h "Output token throughput" "$OUT_DIR"/coloc-C-*.log | awk '{s+=$NF} END{print "    "s}'
kill_all
echo "=== controlled co-location experiment done ==="
