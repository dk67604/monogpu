# Setup — environment & toolchain

> **This does not violate the "no Python by default" rule** (see `../CLAUDE.md`).
> The venv exists to install **reused tools** — the SMG gateway (`smg`) and the
> inference engine (`vllm`), which happen to ship as pip packages — plus lint
> tooling. We still write **no Python source code**. The rule is about not writing
> Python glue by default, not about refusing to install reused binaries.

## Prerequisites (verified on the dev/run box)
- RTX 5080, **sm_120**, 16 GB, **CUDA 13.0** toolkit, Ubuntu 24.04, Python 3.12.
- [`uv`](https://docs.astral.sh/uv/) for fast envs/installs (already present).

## Create the environment
```bash
cd monogpu
uv venv .venv --python 3.12
# dev tooling (lint/hooks):
uv pip install --python .venv/bin/python -r requirements-dev.txt
# reused runtime tools (gateway + engine):
uv pip install --python .venv/bin/python smg vllm
```
`.venv/` is gitignored. Activate with `source .venv/bin/activate` or call binaries
directly (`.venv/bin/smg`, `.venv/bin/shellcheck`, `.venv/bin/pre-commit`).

## What's installed and why
| Package | Role | Ours? |
|---|---|---|
| `smg` | Shepherd Model Gateway — the router we reuse | no (reused) |
| `vllm` | inference engine for the workers | no (reused) |
| `pre-commit` | git hooks (shellcheck etc.) | dev tool |
| `shellcheck-py` | bundles the `shellcheck` binary (no apt/sudo needed) | dev tool |

**Verified working on the RTX 5080 (2026-05-25):** `smg==1.4.1`,
`vllm==0.21.0` with `torch==2.11.0+cu130`. `torch.cuda.is_available()` → True,
device `NVIDIA GeForce RTX 5080`, capability **sm_120**. The cu130 torch wheel
supports sm_120 out of the box (no ptxas workaround needed on this GPU; that's
only for Spark's sm_121). NOTE: the torch wheel is ~2.5 GB — install with
`UV_HTTP_TIMEOUT=600` to avoid the default 30 s timeout.

## Lint locally
```bash
.venv/bin/shellcheck -e SC1091 scripts/*.sh
.venv/bin/pre-commit run --all-files
```

## Verified SMG CLI (smg==1.4.1)
- `smg launch --worker-urls … --policy {random,round_robin,cache_aware,power_of_two,manual} --port …`
  — router only, over workers that already exist (what `scripts/run_gateway.sh` uses).
- `smg serve --backend {sglang,vllm,trtllm} --data-parallel-size N` — launches
  replicas of **one** model **+** router (the "mesh of workers per model" sub-case).
- Pacing/admission knobs available on `launch`: `--rate-limit-tokens-per-second`,
  `--max-concurrent-requests`, `--queue-size`; also `--enable-igw` (multi-model),
  `--pd-disaggregation`, Prometheus metrics.

## Note on sm_121 (DGX Spark) portability
`scripts/env.sh` applies the Triton/ptxas fix needed on sm_121. On this sm_120 box
it's a harmless no-op. CUDA 13 system `ptxas` is present at `/usr/local/cuda/bin`.
