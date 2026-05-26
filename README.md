# monogpu

**Pack many small (≈1B) quantized models onto a single GPU and serve them
efficiently — reusing the [Shepherd Model Gateway (SMG)](https://github.com/lightseekorg/smg)
for routing, and adding only the thin piece SMG lacks: GPU-aware worker
*placement*.**

> Status: **Phase 0 — scaffold.** No engine code yet. See
> [`docs/design.md`](docs/design.md) for the full design and rationale.

## The idea in one paragraph

A router *routes*; it does not *place*. SMG already does multi-backend,
cache-aware routing across vLLM / SGLang / TensorRT-LLM workers — but it assumes
those workers already exist. `monogpu` supplies the missing half: a small,
**shell-first** launcher that probes the GPU, sizes and starts N memory-capped
model workers so they co-fit one GPU (optionally under MPS), and registers them
with SMG. That's the whole MVP — **no Rust, no Python code from us.**

## Why this is interesting on real hardware

The two target GPUs have *opposite* scarce resources, and the design adapts to
whichever bites:

| GPU | Capacity | Bandwidth | Scarce resource |
|---|---|---|---|
| RTX 5080 | 16 GB | 960 GB/s | **capacity** (fits few; runs fast) |
| DGX Spark (GB10) | 128 GB | 273 GB/s | **bandwidth** (fits many; time-share the bus) |

## Quick start (Phase 1 target — not yet implemented)

```bash
scripts/env.sh            # apply sm_121 ptxas fix etc. (see docs)
scripts/mps.sh start      # optional: CUDA MPS for kernel overlap
scripts/launch_workers.sh # probe GPU, start N memory-capped workers, register with SMG
scripts/run_gateway.sh    # run the SMG binary in front of them
```

## Non-negotiables (see [CLAUDE.md](CLAUDE.md))

- **Reuse-first** — prefer SMG / existing engines; justify any from-scratch code.
- **No Python by default** — added only when a concrete need proves it.
- **Optimize with evidence** — every perf change ships with a benchmark delta.
- **Portable** — no hardcoded GPU assumptions; probe at runtime.

## Layout

```
docs/      design, governance, benchmarking
scripts/   shell: env / MPS / worker launcher / gateway runner
bench/     multi-tenant benchmark harness (reuses sibling dgx-spark-benchmark methodology)
```

## License

Apache-2.0 (matches SMG, which this builds around).
