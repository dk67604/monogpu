# CLAUDE.md — monogpu engineering governance

Instructions for any agent (and human) working in this repo. These rules
**override default behavior**. Read them before proposing or writing code.

## What this project is

Pack many small (≈1B) quantized models onto **one GPU** and serve them, by
**reusing the Shepherd Model Gateway (SMG)** for routing and adding only the part
SMG lacks: **GPU-aware worker placement**. The MVP is shell orchestration around
the existing `smg` binary + reused inference engines (vLLM/SGLang/TRT-LLM). See
`docs/design.md` for the full rationale.

## The non-negotiable rules

1. **Reuse-first.** Prefer SMG and existing engines/libraries. Any from-scratch
   code MUST justify, in the PR/commit body, why reuse was insufficient. A router
   already exists (SMG) — do not write one. Inference engines already exist — do
   not write one.
2. **No Python by default.** This repo is **shell-first** for the MVP. Do NOT add
   Python (or a Python toolchain) unless a concrete, benchmarked need requires it
   (e.g. a controller that needs the NVIDIA CUDA Python libs). When that day comes,
   add it deliberately in its own directory with its own toolchain — never as
   default glue.
3. **No new long-lived process unless proven needed.** SMG routes; a bash launcher
   places workers. A bespoke "controller" is **Phase 2+ and evidence-gated** — only
   add it when a benchmark shows a trigger (footprint > VRAM → swapping;
   bandwidth contention hurts SLOs → pacing; multi-tenant SLO → preemption).
4. **Optimize with evidence.** Every performance-motivated change ships with a
   before/after benchmark number in the PR. No speculative optimization.
5. **Portable — never hardcode the GPU.** Probe VRAM / bandwidth / compute
   capability at runtime and adapt. Must work on both RTX 5080 (sm_120, 16 GB,
   capacity-scarce) and DGX Spark (sm_121, 128 GB, bandwidth-scarce).
6. **Thin layer.** Anything we write stays thin and fast; heavy ML stays inside
   the reused engines.

## Toolchains & quality gates

| Language | When | Format / Lint / Test |
|---|---|---|
| **Bash** (now) | MVP | `shellcheck` clean; `set -euo pipefail`; quote everything |
| **Rust** (later) | only when extending SMG / a controller | `cargo fmt`, `clippy -D warnings`, `cargo nextest` |
| **Python** (later) | only when rule 2's bar is met | `ruff` format+lint, `pytest`, `uv` for envs |

- Pre-commit hooks must pass (`.pre-commit-config.yaml`); CI must be green.
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`).
- Work on branches off `main`; do not commit directly to `main` for non-trivial work.
- Commit/push only when the user asks.

## Key external dependency: SMG

- Lives at `../smg` (sibling clone of `lightseekorg/smg`, kept rebased to
  `origin/main`). It is the **routing data plane** — multi-backend, cache-aware.
- SMG `--worker-urls …` routes to workers **that already exist**; it does not
  start them or manage GPU memory. That gap is this repo's whole job.
- SMG's `mesh` (`--ha-mesh`) is **cross-node HA gossip**, NOT an intra-GPU mesh.
- Its CLI/features move fast (we just rebased +522 commits). **Always verify the
  current `smg --help` / `../smg/docs`** before relying on a flag — do not trust
  stale notes.

## Repo layout

```
docs/      design.md (authoritative design), governance.md, benchmarking.md
scripts/   env.sh (sm_121 ptxas fix), mps.sh, launch_workers.sh, run_gateway.sh
bench/     multi-tenant harness; reuses sibling dgx-spark-benchmark methodology + power monitor
```

## How to run / verify (Phase-1 target)

1. `scripts/env.sh` — sets sm_121 Triton/ptxas env (see `docs/design.md`).
2. `scripts/launch_workers.sh` — probe GPU → size & start N memory-capped workers
   (+ optional MPS) → register with SMG.
3. `scripts/run_gateway.sh` — run the `smg` binary in front.
4. `bench/` — scale 1→5 workers; measure aggregate throughput, per-model
   p50/p90/p99, interference, bandwidth utilization, energy.

## Definition of done (any change)

- shellcheck/lint/tests green; docs updated; if perf-related, a benchmark delta is
  included; no hardcoded GPU assumptions introduced; reuse-first justified.
