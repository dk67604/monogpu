#!/usr/bin/env bash
# Start/stop NVIDIA CUDA MPS (Multi-Process Service) so co-located worker
# processes can overlap kernels on the one GPU. Soft sharing only (no MIG on
# consumer Blackwell). Usage: scripts/mps.sh {start|stop|status}
#
# NOTE: verify MPS is supported on the target (works on most CUDA GPUs; confirm on
# sm_121). MPS does not add bandwidth — it overlaps compute (e.g. a prefill on one
# worker with a decode on another).
set -euo pipefail

cmd="${1:-status}"

case "$cmd" in
  start)
    if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "[mps] already running"
    else
      nvidia-cuda-mps-control -d
      echo "[mps] started"
    fi
    ;;
  stop)
    echo quit | nvidia-cuda-mps-control 2>/dev/null || echo "[mps] not running"
    echo "[mps] stopped"
    ;;
  status)
    if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "[mps] running"
    else
      echo "[mps] not running"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
