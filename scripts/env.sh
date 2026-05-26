#!/usr/bin/env bash
# Environment setup. SOURCE this (don't execute): `source scripts/env.sh`
#
# Sets up, in order:
#   (1) the project venv  → puts vllm / smg / ninja on PATH
#   (2) the CUDA toolkit   → flashinfer JIT-compiles kernels at first run and
#                            needs `nvcc` on PATH (this was the root cause of an
#                            early "FileNotFoundError: 'ninja'/nvcc" worker crash)
#   (3) the sm_121 Triton/ptxas workaround — applied ONLY on Blackwell sm_121
#       (DGX Spark). On sm_120 (RTX 5080) torch's cu130 wheel handles the arch
#       natively, so forcing 12.1 would be wrong — hence the runtime GPU probe.
#       Ref: https://github.com/triton-lang/triton/issues/10331
# Opt out of the sm_121 fix with: export DGX_SKIP_SM121_FIX=1

# Resolve repo root from this file's location (works when sourced).
_env_src="${BASH_SOURCE[0]:-$0}"
_repo="$(cd "$(dirname "$_env_src")/.." && pwd)"

# (1) project venv — vllm, smg, ninja
if [ -f "$_repo/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$_repo/.venv/bin/activate"
fi

# (2) CUDA toolkit on PATH (nvcc for flashinfer JIT)
if [ -d /usr/local/cuda/bin ]; then
  case ":$PATH:" in
    *":/usr/local/cuda/bin:"*) ;;
    *) export PATH="/usr/local/cuda/bin:$PATH" ;;
  esac
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

# (3) sm_121-only Triton/ptxas workaround (probe the GPU first)
_cc=""
if command -v nvidia-smi >/dev/null 2>&1; then
  _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi
if [ "${DGX_SKIP_SM121_FIX:-0}" != "1" ] && [ "$_cc" = "12.1" ]; then
  : "${TORCH_CUDA_ARCH_LIST:=12.1+PTX}"
  export TORCH_CUDA_ARCH_LIST
  if [ -z "${TRITON_PTXAS_PATH:-}" ] && [ -x /usr/local/cuda/bin/ptxas ]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
  fi
  unset TRITON_OVERRIDE_ARCH 2>/dev/null || true
  echo "[env] sm_121 detected → applied Triton/ptxas workaround"
fi

echo "[env] venv=${VIRTUAL_ENV:-<none>} cuda=${CUDA_HOME:-<none>} compute_cap=${_cc:-?}"
