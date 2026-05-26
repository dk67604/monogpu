#!/usr/bin/env bash
# Run the REUSED Shepherd Model Gateway (SMG) in front of the workers placed by
# launch_workers.sh. We write no router — this just invokes the existing binary.
#
# Uses the `smg launch` subcommand (router only, over existing worker URLs).
# Verified against smg==1.4.1: `smg launch --worker-urls … --policy … --port …`.
# Policies: random | round_robin | cache_aware | power_of_two | manual.
# SMG also has admission/pacing knobs we can set later (see PACING_ARGS below):
#   --rate-limit-tokens-per-second, --max-concurrent-requests, --queue-size.
#
# Env:
#   WORKERS_FILE   file of worker URLs (one per line)   [scripts/.workers]
#   POLICY         SMG routing policy                   [cache_aware]
#   GATEWAY_PORT   SMG listen port                      [30000]
#   SMG_BIN        smg binary                           [smg]
#   PACING_ARGS    extra flags (e.g. --rate-limit-tokens-per-second N)  []
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=scripts/env.sh
. "$SCRIPT_DIR/env.sh"   # venv + CUDA on PATH (smg lives in the venv); before strict mode
set -euo pipefail

WORKERS_FILE="${WORKERS_FILE:-scripts/.workers}"
POLICY="${POLICY:-cache_aware}"
GATEWAY_PORT="${GATEWAY_PORT:-30000}"
SMG_BIN="${SMG_BIN:-smg}"
PACING_ARGS="${PACING_ARGS:-}"

if ! command -v "$SMG_BIN" >/dev/null 2>&1; then
  echo "[gateway] '$SMG_BIN' not found. Install SMG: pip install smg (into the venv)" >&2
  exit 1
fi
if [ ! -s "$WORKERS_FILE" ]; then
  echo "[gateway] no workers in '$WORKERS_FILE' — run scripts/launch_workers.sh first" >&2
  exit 1
fi

# Read worker URLs (one per line) into an array.
mapfile -t workers < "$WORKERS_FILE"
echo "[gateway] routing ${#workers[@]} worker(s) | policy=${POLICY} | port=${GATEWAY_PORT}"

# shellcheck disable=SC2086  # PACING_ARGS intentionally word-split
exec "$SMG_BIN" launch \
  --worker-urls "${workers[@]}" \
  --policy "$POLICY" \
  --port "$GATEWAY_PORT" \
  $PACING_ARGS
