# Benchmarking — multi-tenant, single-GPU

Single-model latency/throughput is necessary but not sufficient. This project lives
or dies on **multi-tenant** behavior, so we measure the whole gateway under load.

Reuse, don't rebuild: the sibling repo `../dgx-spark-benchmark` already has a sound
measurement methodology (greedy two-pass TTFT/decode, token accounting, percentiles)
and a **power monitor** (pynvml/nvidia-smi). `bench/` here drives the *gateway* and
reuses that methodology rather than re-deriving it.

## Metrics (beyond single-model TTFT/ITL/throughput)
- **Aggregate vs per-model throughput** — confirm aggregate decode tok/s is
  bus-capped on Spark regardless of model count; capacity-bound (not bus-bound) on
  the 5080.
- **Per-model latency percentiles** (p50/p90/p99) **under concurrent load**.
- **Interference factor** = latency(model A alone) ÷ latency(A while B,C… run).
  Quantifies the no-MIG soft-sharing penalty.
- **Bandwidth utilization** — measured GB/s vs the GPU's ceiling (roofline attainment).
- **Goodput / SLO attainment** — % requests meeting TTFT + p99 ITL targets.
- **Router overhead** — added latency from SMG vs talking to a worker directly.
- **Energy / tokens-per-joule** — via the sibling power monitor.
- **Warm-pool hit rate & cold-start cost** — once swapping exists (Phase 2).

## Workloads
- **Capacity-vs-throughput curve:** scale 1→5 co-located 1B workers; plot
  aggregate throughput + per-model latency.
- **Multi-tenant mix:** N models, each a Poisson arrival process + bursts.
- **Request-rate (QPS) sweep** per model → load-vs-latency knee.
- **Shared-prefix** workload → exercises SMG cache-aware routing (hit-rate delta
  vs round_robin).
- **Skewed popularity** (Zipfian model choice) → stresses the warm pool (Phase 2).

## Procedure notes
- Compare **routing policies** (random / round_robin / power_of_two / cache_aware)
  and **layouts** (N models × M replicas; MPS on/off).
- Sweep **concurrency** to find the saturation knee (expect a fast plateau on Spark).
- Always report percentiles under load, not means.
- Run the same suite on **both** GPUs; confirm conclusions flip with the scarce
  resource.

## Prefix caching — two distinct levers

1. **vLLM automatic prefix caching (APC), inside one worker.** Skips re-prefilling
   a shared prefix. Enabled by default (`enable_prefix_caching=True`). A
   *random*-prompt benchmark hides it entirely (no shared prefix). Measure with
   `bench/prefix_ab.sh` (random vs `prefix_repetition`). Measured on the RTX 5080
   (0.5B): negligible with a short prefix + decode-heavy run, but **~2× throughput
   and ~−64% P99 TTFT** with a 2048-token shared prefix + 32-token output
   (prefill-dominated). Payoff ∝ shared-prefix length and prefill share.
2. **SMG cache-aware *routing*, across replicas.** Only matters with **multiple
   replicas of one model** — the router (radix-tree of observed prefixes) sends
   same-prefix requests to the replica likeliest to have them cached. With one
   replica per model it is a no-op.
   **Measured** with `bench/routing_ab.sh` (2× Qwen2.5-0.5B, 2048-tok prefixes,
   KV constrained to `gmu=0.30`, cold cache per policy), sweeping the working set:

   | prefixes | cache_aware | round_robin | winner |
   |---|--:|--:|---|
   | 8 (fits everywhere)      | 62.4 req/s | 60.6 | tie |
   | 128 (halved set fits)    | 55.2 | 53.4 | cache_aware **+3.5%** |
   | 256 (exceeds both)       | 38.5 | 46.3 | round_robin **+20%** |

   **Conclusion: cache-aware routing is a narrow, regime-dependent optimization,
   and on small models it is *not* worth it.** It only edges ahead when the working
   set is between 1× and 2× a replica's cache (pinned half fits, full set doesn't);
   below that it ties, and *above* it it LOSES — pinning sacrifices load balance
   while the cache benefit vanishes (each replica's pinned half also overflows).
   On a 0.5B, prefill is cheap, so cache savings barely beat the load-balance
   penalty (best ~3.5%). SGLang's reported ~1.9× comes from **large models**
   (expensive prefill) + high reuse — unreachable with 2 co-resident replicas on
   16 GB. **Takeaway for the 5080 with small models: prefer `round_robin` /
   `power_of_two`; reserve `cache_aware` for large-model / high-reuse serving on
   bigger GPUs.**

## Controlled co-location experiment (bench/colocation_ab.sh)

Holding benchmark concurrency (48 total) and CUDA graphs (ON) constant, varying
only the GPU-memory split, direct-to-worker (no gateway):

| config | gmu | 0.5B output throughput |
|---|---|--:|
| A — dedicated | 0.85 | 12,766 tok/s |
| B — memory-split | 0.30 | 12,838 tok/s |

**Memory split has no effect** at this concurrency — KV cache wasn't the bottleneck.
**Not cleanly measured:** the 3-model *aggregate* under contention — config C's third
worker OOM'd (3 models + CUDA graphs don't fit 16 GB), so no contention number is
claimed. Earlier cross-config "concurrency wins" comparisons were **confounded** (the
single-model baseline used different concurrency + CUDA graphs off) and are retracted.

> Cross-config caveat: the prefix-cache and routing A/Bs ran with CUDA graphs OFF
> (`--enforce-eager`), so their absolute tok/s are not comparable to the table above;
> only their *relative* effects are results.

## Tooling candidates
SMG's own benchmark tooling, NVIDIA GenAI-Perf, or an async request-rate driver
that reuses the sibling repo's harness. Decide when `bench/` is implemented.
