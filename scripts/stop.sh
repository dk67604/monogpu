#!/usr/bin/env bash
# Stop all workers launched by launch_workers.sh (PIDs in run/worker.pids) and any
# running SMG gateway. Best-effort; safe to run repeatedly.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../run}"
PIDS_FILE="${PIDS_FILE:-${RUN_DIR}/worker.pids}"

if [ -f "$PIDS_FILE" ]; then
  while read -r pid; do
    [ -z "$pid" ] && continue
    if kill "$pid" 2>/dev/null; then
      echo "[stop] killed worker pid $pid"
    fi
  done < "$PIDS_FILE"
  rm -f "$PIDS_FILE"
fi

# Gateway (and any stragglers) by name. The running SMG process is named
# "smg::router" (NOT "smg launch"), so match that too. Safe from self-match: this
# script's own cmdline is "bash .../stop.sh", which does not contain the patterns.
for _pat in "smg::router" "smg launch" "vllm serve"; do
  pkill -f "$_pat" 2>/dev/null && echo "[stop] killed: $_pat" || true
done
echo "[stop] done"
