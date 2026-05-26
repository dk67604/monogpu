# Governance — the *why* behind the rules

The enforceable rules live in [`../CLAUDE.md`](../CLAUDE.md). This file records the
reasoning, so future contributors understand intent, not just the letter.

## Reuse-first
A router and inference engines already exist and are mature (SMG, vLLM, SGLang,
TRT-LLM). Re-implementing them would be a large, low-value maintenance burden that
drifts from upstream. Our value is the **one missing piece** (GPU-aware placement)
and the **integration/benchmarking** — not reinvention. Any from-scratch code must
name what existing thing it considered and why it fell short.

## No Python by default
Two reasons. (1) The hot path (routing) is latency-sensitive and Python's GIL is a
known bottleneck for routers — which is why SMG is Rust. (2) Scope discipline: a
control loop is easy to start in Python and hard to keep thin. We stay shell-first
for the MVP and add Python only when a concrete, benchmarked need (e.g. a controller
using NVIDIA CUDA Python libs) justifies it — then deliberately, in its own dir.

## Optimize with evidence
This is a performance project on bandwidth-constrained hardware. Intuition about
GPU performance is frequently wrong; the roofline is non-obvious. Every perf change
must carry a measured before/after, produced by `bench/`.

## Portable, never hardcode the GPU
The two target GPUs have *opposite* scarce resources (5080 capacity-bound, Spark
bandwidth-bound). A design tuned to one silently fails the other. Probe at runtime
and let the scarce resource pick the strategy.

## Evidence-gated escalation
MVP = SMG + bash launcher. We escalate to a bespoke controller (and to Rust/Python)
**only when a benchmark exhibits a trigger** (footprint > VRAM, bandwidth-induced
SLO miss, multi-tenant priority). This keeps complexity proportional to demonstrated
need.
