# monogpu — design

Authoritative design doc. Built from a pre-code design dive; see the sibling
`dgx-spark-benchmark/research/multi-model-gateway-dgx-spark.md` for the deeper
literature survey and roofline analysis.

## 1. Goal

Serve **many small (≈1B) quantized models on a single GPU**, maximizing
utilization, behind one OpenAI-compatible endpoint — by **reusing the Shepherd
Model Gateway (SMG)** and adding only **GPU-aware worker placement**.

## 2. The core insight: route ≠ place

A router distributes *requests* to workers that **already exist**. SMG
(`smg --worker-urls …`) is exactly that: multi-backend (vLLM / SGLang /
TensorRT-LLM / cloud), cache-aware (router-side prefix radix tree, backend-agnostic),
8 routing policies, gRPC, failover. **What no router does** is *start* the workers
or decide how GPU memory is divided. That placement step is the entire reason this
project exists.

So: **SMG (reused) = data plane. `monogpu` = a thin placement layer + (later)
resource-aware control.**

## 3. Hardware reality (drives everything)

| GPU | Capacity | Bandwidth | sm | Scarce resource | Strategy |
|---|---|---|---|---|---|
| RTX 5080 | 16 GB | 960 GB/s | sm_120 | **capacity** | fit few, run concurrently |
| DGX Spark GB10 | 128 GB | 273 GB/s | sm_121 | **bandwidth** | fit many warm, time-share bus |

- Decode is **memory-bandwidth-bound** (reads all weights per token). On the 5080
  bandwidth is plentiful → concurrency scales until VRAM fills. On Spark the 273
  GB/s bus is the bottleneck → many models fit but their *aggregate* decode rate is
  bus-capped; keep a warm pool and time-multiplex.
- 1B models are tiny enough to be **kernel-launch / overhead-bound**, which makes
  per-request overhead (routing, dispatch) the thing to keep small — hence reusing
  SMG's native-Rust, sub-ms routing rather than writing our own.
- **No MIG** on consumer Blackwell (5080 or Spark). GPU sharing = **CUDA MPS** or
  streams or time-slicing — soft sharing, no hard isolation, so the launcher must
  budget memory itself.
- **sm_121 caveat:** PyTorch's bundled `ptxas` predates sm_121; Triton kernels fail
  with "no kernel image…". Fix via `scripts/env.sh`
  (`TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, `TORCH_CUDA_ARCH_LIST=12.1+PTX`,
  unset `TRITON_OVERRIDE_ARCH`). Ref: triton-lang/triton#10331.

## 4. Architecture

### MVP (Phase 1) — shell only, no Rust/Python from us
```
 scripts/launch_workers.sh (one-shot)
   ├─ probe GPU: VRAM, bandwidth, compute capability
   ├─ compute per-worker memory cap so N workers co-fit
   ├─ start N engine workers (vLLM/SGLang) on the one GPU (+ optional MPS)
   └─ register their URLs with SMG
                    │
 scripts/run_gateway.sh → smg --worker-urls … --policy cache_aware   (REUSED binary)
                    │  OpenAI endpoint
            clients ┘
```
Nothing here is novel except the launcher. SMG does ingress, routing, balancing.

### Phase 2+ (evidence-gated only) — resource-aware controller
Add a controller **above** SMG **only when a benchmark proves a trigger**:
- Σ model footprint > VRAM → on-demand load/evict (warm-pool swapping).
- bandwidth contention hurts SLOs (Spark) → bandwidth-aware pacing / concurrency cap.
- multi-tenant SLO/priority → preemption (Shepherd-style).

**Verified against `smg==1.4.1` (see `setup.md`):** SMG already exposes pacing/
admission knobs on `smg launch` — `--rate-limit-tokens-per-second`,
`--max-concurrent-requests`, `--queue-size` — plus `--enable-igw` (multi-model),
`--pd-disaggregation`, and `smg serve --backend {sglang,vllm,trtllm}
--data-parallel-size N` (single-model replica launching). So a future "controller"
mostly **computes good values and sets these flags from the GPU probe**, rather
than implementing pacing from scratch. What's genuinely absent: *automatic*
bandwidth-awareness (knowing the bus ceiling), per-worker VRAM placement for N
*different* models, and model-residency swapping. Language decided then; not before.

## 5. What we explicitly do NOT build

- A router (SMG exists). - An inference engine (vLLM/SGLang/TRT-LLM exist).
- A Python service by default. - A controller before evidence demands it.
- Anything assuming a specific GPU.

## 6. "Mesh" clarification

SMG's `mesh` (`--ha-mesh`, SWIM gossip) is **cross-node high availability**, not an
intra-GPU worker mesh. "A mesh of workers per model" here simply means several
worker replicas of one model on the GPU, registered as multiple `--worker-urls`,
which SMG load-balances (cache-aware / power-of-two).

## 7. Models

Target ≈1B quantized: Llama-3.2-1B, Qwen3-0.6B/1.7B, SmolLM2-1.7B, Gemma-3-1B.
Keep model size a **parameter**, not an assumption. Quantization: AWQ/GPTQ (INT4),
FP8, or Blackwell-native NVFP4 (stretches both VRAM footprint and decode speed).

## 8. Open items (decide as we build)
- Final project name; exact SMG invocation (verify against rebased `../smg`).
- Whether `bench/` reuses the sibling repo in place or vendors its methodology.
- MPS on/off default per GPU (verify MPS works on sm_121).

## References
- SMG: https://github.com/lightseekorg/smg  (local clone: `../smg`)
- sm_121 ptxas fix: https://github.com/triton-lang/triton/issues/10331
- Shepherd (NSDI'23): https://www.usenix.org/system/files/nsdi23-zhang-hong.pdf
- Roofline / bandwidth-bound decode: https://arxiv.org/html/2402.16363v4
- Sibling research doc: `../dgx-spark-benchmark/research/multi-model-gateway-dgx-spark.md`
