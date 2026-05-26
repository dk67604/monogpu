---
title: "Serving a Fleet of LLMs on One RTX 5080: Multi-Model on a Single Consumer GPU"
published: false
description: "Serving several chat LLMs from one 16 GB RTX 5080 by reusing an existing router + ~150 lines of shell — with controlled benchmarks: where memory split doesn't matter, where prefix caching doubles throughput, and where 'smart' routing made things slower."
tags: machinelearning, ai, llm, performance
# cover_image: https://raw.githubusercontent.com/dk67604/monogpu/main/docs/img/memory_split.png
---

*Every number below was measured on a single RTX 5080 (16 GB) and is reproducible from the repo. Each result states the exact config it was measured under; I don't compare numbers across configs, and I flag anything we did **not** cleanly measure.*

## TL;DR

You can serve **several small chat LLMs from one 16 GB RTX 5080**, behind a single OpenAI-compatible endpoint, by **reusing an existing router** (the Shepherd Model Gateway) and **vLLM** for the workers — the only thing I wrote is **~150 lines of shell** to place and route the models. Three controlled findings:

- **A 0.5B model serves ~12,800 tok/s** (CUDA graphs on, concurrency 48).
- **Giving a model 1/3 of the GPU's memory instead of all of it made no difference** to throughput (12,766 vs 12,838 tok/s).
- **Prefix caching doubled throughput** on a prefill-heavy workload; **cache-aware routing *lost* 20%** to plain round-robin in one regime.

## Why this works on a 16 GB card

Small models are tiny: a quantized ~1B model is roughly 0.5–2 GB. So several fit in 16 GB at once, and you can serve them concurrently behind one endpoint. The only real question is *how* to place and route them cleanly on one card.

## The key idea: a router routes, it does not *place*

Mature routers already exist (NVIDIA Dynamo, vLLM's router, the Shepherd Model Gateway). They give you an OpenAI endpoint and cache/load-aware routing across workers — but they **route to workers that already exist**; they don't start the workers or divide GPU memory. So "many models on one GPU" only needs a **placement** step: start N model servers, each memory-capped to co-fit, and register them. That's a shell script:

```
clients → SMG (reused binary) → N vLLM workers on one GPU
            ↑ a ~15-line bash launcher places + registers the workers
```

I ran **three chat models** — Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-360M-Instruct — co-resident behind one gateway, each reachable by model name.

## Three gotchas we actually hit on the 5080

1. **flashinfer's JIT needs `ninja` and `nvcc` on PATH.** Launch a vLLM worker from a bare venv path and it dies with `FileNotFoundError: 'ninja'` inside kernel compilation. Activate the venv *and* put the CUDA toolkit on PATH first.
2. **Don't start co-located workers concurrently.** vLLM measures `--gpu-memory-utilization` against *total* memory at startup; launch two at once and they race → one gets "No available memory for the cache blocks." Start them **sequentially** (health-check each before the next).
3. **Three models + CUDA graphs don't fit 16 GB.** With graphs on, the third worker OOM'd. Co-locating 3 means running with graphs off (`--enforce-eager`) or fewer models.

## Finding 1: the memory split didn't matter (controlled)

I gave one 0.5B the whole GPU vs. just 30% of it, holding everything else constant (concurrency 48, CUDA graphs on):

![Memory split has no effect](https://raw.githubusercontent.com/dk67604/monogpu/main/docs/img/memory_split.png)

**12,766 vs 12,838 tok/s — identical.** At this concurrency the KV cache wasn't the bottleneck, so shrinking it to make room for neighbors cost nothing. Good news for co-location: you can hand most of the card to other models without hurting a small model's throughput.

## Finding 2: prefix caching doubled throughput (when prefill dominates)

vLLM's automatic prefix caching is on by default, but a random-prompt benchmark *hides* it (no shared prefix). Same model, same config (graphs off), only the workload's shared-prefix fraction changes:

![Prefix cache A/B](https://raw.githubusercontent.com/dk67604/monogpu/main/docs/img/prefix_cache.png)

With a 2048-token shared prefix and 32-token output (prefill-heavy), throughput **doubled (1,153 → 2,316 tok/s)** and p99 TTFT dropped **64%**. With a short prefix and long output it was a modest ~15%. So a shared system prompt / RAG context is worth a lot; random prompts get nothing.

## Finding 3: "smart" routing isn't always smart

Cache-aware routing only matters with *multiple replicas* of one model (pin same-prefix requests to the same replica). Holding config constant and sweeping the prefix working set against constrained per-replica caches:

![Routing regimes](https://raw.githubusercontent.com/dk67604/monogpu/main/docs/img/routing_regimes.png)

| Prefix working set | cache_aware | round_robin | result |
|---|--:|--:|---|
| small (fits everywhere) | 62.4 | 60.6 req/s | tie |
| sweet spot | 55.2 | 53.4 | +3.5% (within noise) |
| oversized | 38.5 | 46.3 | **round_robin +20%** |

When the working set overflows a replica's cache, cache-aware pinning sacrifices load balance and **loses by 20%**. On these small models, plain round-robin / power-of-two was the better default.

## What we did NOT measure cleanly

To be straight about the limits:

- **The multi-model *aggregate* throughput under contention.** Our controlled 3-model run had a worker OOM (graphs on, 16 GB), so we don't have a clean contention number — it's omitted rather than guessed.
- **One GPU.** Everything here is the RTX 5080; we make no claims about other hardware.
- The prefix-cache and routing runs used CUDA graphs **off**, so their absolute tok/s aren't comparable to Finding 1's — only the *relative* effects (2×, −20%) are the result.

## Reproduce it

```bash
scripts/launch_workers.sh   # probe GPU, size + start N capped workers (sequential)
scripts/run_gateway.sh      # smg launch in front
bench/sweep.sh              # QPS + goodput sweep
bench/chart.sh              # self-contained HTML report
```

## Closing

The takeaway isn't a throughput record — it's that **reuse + a shell script** gets a working multi-model serving stack onto a consumer GPU, and that controlled measurement beats intuition: memory split didn't matter, prefix caching was a 2× win *only* with shared prefixes, and cache-aware routing *lost* in the wrong regime. Measure your own workload.

*Repo, scripts, and raw benchmark JSON: **https://github.com/dk67604/monogpu**.*
