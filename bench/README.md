# bench/

Multi-tenant, single-GPU benchmark harness for the gateway. See
[`../docs/benchmarking.md`](../docs/benchmarking.md) for the full metric list and
methodology.

**Reuse, don't rebuild.** The sibling repo `../../dgx-spark-benchmark` already has a
validated measurement methodology (two-pass TTFT/decode, token accounting,
percentiles) and a power monitor. This harness drives the *gateway* under
multi-tenant load and reuses that methodology rather than re-deriving it.

Status: **placeholder** (Phase 1). To be implemented after the launcher + gateway
MVP runs end-to-end. First experiment: scale 1→5 co-located 1B workers; plot
aggregate throughput + per-model p50/p90/p99 latency + interference factor on both
the RTX 5080 and DGX Spark.
