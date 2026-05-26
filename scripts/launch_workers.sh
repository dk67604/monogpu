#!/usr/bin/env bash
# Place N memory-capped model workers on ONE GPU and record their URLs for the
# gateway. This is the one piece SMG does not do (a router routes; it does not
# place). MVP: vLLM workers; SGLang/TRT-LLM are drop-in later.
#
#   MODELS="meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3-1.7B" scripts/launch_workers.sh
#
# Env:
#   MODELS         space-separated HF model ids (one worker each)        [required]
#   BASE_PORT      first worker port, incremented per model              [8001]
#   MEM_HEADROOM   fraction of VRAM reserved (OS/CUDA ctx/other)         [0.15]
#   WORKERS_FILE   where worker URLs are written for run_gateway.sh      [scripts/.workers]
#   EXTRA_ARGS     extra flags passed verbatim to `vllm serve`           []
#
# TODO(verify): confirm `vllm serve` flags and the SMG registration path against
# the installed versions (vLLM and ../smg are fast-moving). cargo/vllm are not
# assumed present on the dev box — run this on the GPU box.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=scripts/env.sh
. "$SCRIPT_DIR/env.sh"   # venv + CUDA on PATH (+ sm_121 fix); before strict mode
set -euo pipefail

: "${MODELS:?set MODELS to a space-separated list of HF model ids}"
BASE_PORT="${BASE_PORT:-8001}"
MEM_HEADROOM="${MEM_HEADROOM:-0.15}"
WORKERS_FILE="${WORKERS_FILE:-scripts/.workers}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_DIR="${RUN_DIR:-run}"
PIDS_FILE="${PIDS_FILE:-${RUN_DIR}/worker.pids}"
WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-180}"
# Small co-located models: --enforce-eager skips slow/memory-hungry CUDA-graph
# capture (~40s/worker). Recommended on by default; set ENFORCE_EAGER=0 to disable.
[ "${ENFORCE_EAGER:-1}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --enforce-eager"

# --- probe GPU -------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[launch] nvidia-smi not found — need a GPU host" >&2
  exit 1
fi
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
echo "[launch] GPU: ${gpu_name} | VRAM: ${vram_mib} MiB"

# --- compute per-worker memory fraction so N workers co-fit ----------------
# Each vLLM instance interprets --gpu-memory-utilization as a fraction of TOTAL
# VRAM. With N co-located workers we split (1 - headroom) evenly. Quantize models
# so weights fit inside their slice.
read -r -a model_arr <<< "$MODELS"
n="${#model_arr[@]}"
frac="$(awk -v n="$n" -v h="$MEM_HEADROOM" 'BEGIN{printf "%.3f", (1.0-h)/n}')"
echo "[launch] ${n} worker(s); per-worker --gpu-memory-utilization=${frac}"

mkdir -p "$RUN_DIR"
: > "$WORKERS_FILE"   # truncate
: > "$PIDS_FILE"
port="$BASE_PORT"
for model in "${model_arr[@]}"; do
  url="http://localhost:${port}"
  logf="${RUN_DIR}/worker-${port}.log"
  echo "[launch] starting worker: ${model} on ${url} (gmu=${frac}) -> ${logf}"
  # setsid detaches the worker into its own session so it survives this script
  # exiting. < /dev/null so it never blocks on stdin.
  # shellcheck disable=SC2086  # EXTRA_ARGS intentionally word-split
  setsid env CUDA_VISIBLE_DEVICES=0 vllm serve "$model" \
    --port "$port" \
    --gpu-memory-utilization "$frac" \
    --trust-remote-code \
    $EXTRA_ARGS > "$logf" 2>&1 < /dev/null &
  echo "$!" >> "$PIDS_FILE"
  echo "$url" >> "$WORKERS_FILE"
  # SEQUENTIAL startup: wait for THIS worker to be healthy before launching the
  # next. Co-located workers must NOT profile GPU memory concurrently — that
  # races and yields "No available memory for the cache blocks" (vLLM measures
  # --gpu-memory-utilization against TOTAL memory at startup).
  echo "[launch] waiting for ${url}/health (up to ${WORKER_READY_TIMEOUT}s)..."
  _ok=0
  for _t in $(seq 1 "$WORKER_READY_TIMEOUT"); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "${url}/health" 2>/dev/null || echo 000)" = "200" ]; then
      echo "[launch]   ${url} ready"; _ok=1; break
    fi
    sleep 2
  done
  [ "$_ok" = "1" ] || echo "[launch]   WARNING: ${url} not healthy yet — see ${logf}"
  port=$((port + 1))
done

echo "[launch] ${n} worker(s) launched (PIDs in ${PIDS_FILE}); URLs in ${WORKERS_FILE}:"
cat "$WORKERS_FILE"
echo "[launch] tail logs: tail -f ${RUN_DIR}/worker-*.log"
echo "[launch] when healthy: scripts/run_gateway.sh   | stop all: scripts/stop.sh"
