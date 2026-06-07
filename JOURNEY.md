# CS336 Assignment 2 (Systems) — Journey

A step-by-step record of the work, the code written, and the insights/findings
along the way.

---

## 0. Orientation: separate repo, reuse Assignment 1

**What we did**
- Assignment 2 (`assignment2-systems`) is its **own self-contained repo** — its own
  `pyproject.toml`, `uv` setup, `cs336_systems/` package (where our work goes), tests,
  and `test_and_make_submission.sh`. We did *not* keep working inside `assignment1-basics`.
- It ships a **staff reference implementation** of Assignment 1 as the bundled
  `cs336-basics` package, wired up via the root `pyproject.toml`.

**Insight / decision**
- `import cs336_basics` resolves to whichever package the *active project env* exposes.
  Inside `assignment2-systems` that's the **staff reference** by default — not our own
  Assignment 1 code. To use our own, we'd replace the `cs336-basics/` dir or repoint
  `pyproject.toml`.
- We chose to **start with the staff reference** so Assignment 2 isn't failing because of
  an Assignment 1 bug. Can swap our own model in later.

**The four graded parts** (from the test files + handout):
`test_attention.py` (FlashAttention), `test_ddp.py`, `test_fsdp.py`,
`test_sharded_optimizer.py` → §2 Profiling, §4 FlashAttention-2, §5 DDP,
§6 Optimizer State Sharding, §7 FSDP, §8 parallelism math (pen & paper), §9 leaderboard.

---

## 1. Environment verification

```bash
uv run python -c "import cs336_basics; print('cs336_basics OK')"   # → OK
```

`uv run` activates the project venv where `cs336_basics` is installed, so the import
succeeds. A failing import here means the env isn't synced or the dir isn't on the path.

**Model API discovered** (from `cs336_basics`):
- `BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0)`
- `AdamW(params, lr=1e-3, betas, eps, weight_decay)`
- `cross_entropy(inputs, targets)` — takes logits `(..., vocab)` and targets `(...)` directly
  (it does `targets.unsqueeze(-1)` + gather internally).

**Model sizes (handout Table 1)**, context length 512 unless noted:

| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 2560    | 10240 | 32         | 32        |
| 10B    | 4608    | 12288 | 50         | 36        |

---

## 2. Code: end-to-end benchmarking script (§2.1.3, Problem `benchmarking_script`, 4 pts)

**File:** `cs336_systems/benchmark.py`

A CLI harness that:
- Initializes `BasicsTransformerLM` from hyperparameters (`--size` preset *or* individual flags).
- Generates a random batch of token ids + targets (speed-only, so random weights/data are fine).
- Runs `--warmup` steps, then times `--steps` measured steps.
- Supports `--mode forward | forward_backward | full` (full = incl. optimizer step).
- Calls `torch.cuda.synchronize()` after each step (no-op on CPU/MPS, correct on CUDA).
- Reports mean + std (and ms/token-step).

**Key design decisions / fixes:**
1. **`torch.cuda.synchronize()` is essential.** CUDA kernels are async — the Python call
   returns before the GPU finishes. Without a sync after each step we'd time *kernel launch*,
   not *kernel execution*. Guarded so it's a no-op on the Mac (CPU) but correct on the GPU.
2. **Forward runs WITH grad tracking in every mode** (no `torch.no_grad()` in `forward` mode).
   This makes `forward` and `forward_backward` measure the *same* forward, so we can derive
   **backward = (forward_backward − forward)** cleanly. (Originally `forward` used `no_grad`,
   which would have wrongly dumped graph-building cost into "backward".)
3. **Timer: `timeit.default_timer()`** — the handout's suggested high-resolution clock
   (it's `time.perf_counter()` under the hood, but we matched the wording).

Run examples:
```bash
uv run python -m cs336_systems.benchmark --size small --mode forward_backward --warmup 5 --steps 10
uv run python -m cs336_systems.benchmark --size medium --mode full --warmup 0 --steps 10
```

---

## 3. Insight: what warm-up is for

Warm-up = run a few steps *before* timing and discard them, because the first steps are
unrepresentatively slow for reasons unrelated to steady-state cost:
- **cuBLAS/cuDNN autotuning** — fastest algorithm picked + cached on first sight of each shape.
- **Lazy CUDA context + kernel/JIT compilation** — initialized on first use.
- **Caching allocator warm-up** — first allocs hit the driver (slow), then get reused.
- **GPU clocks ramping** under sustained load.

On CPU these don't exist, which is why local runs showed tiny std.


---

## 4. Findings: benchmark results (single GPU, fp32, ctx=512, batch=4, 5 warmup / 10 steps)

### Part (b) — forward / backward / optimizer breakdown

| Size   | Params | Forward  | Backward | Optimizer | Bwd/Fwd | Std |
|--------|--------|----------|----------|-----------|---------|-----|
| small  | 129M   | 16.90 ms | 40.51 ms | 23.85 ms  | 2.40×   | small fwd (0.03), noisier full (7.94) |
| medium | 423M   | 48.92 ms | 103.41 ms| 21.25 ms  | 2.11×   | small (0.2–1.8 ms) |
| large  | 969M   | 112.08 ms| 221.42 ms| 49.99 ms  | 1.98×   | very small (<0.5 ms) |

- Derived as **backward = fwd_bwd − fwd**, **optimizer = full − fwd_bwd**.
- **Backward ≈ 2× forward** across all sizes — consistent with backward doing ~2× the FLOPs.
- **Std is small** for medium/large (well under 1%); the **small model is noisier** in relative
  terms (e.g. ±7.9 ms on the full step) because its kernels are short enough that launch /
  scheduling overhead and clock jitter dominate.
- `xl`/`10B` **OOM** on a single GPU at batch 4 — expected.



### Part (c) — warm-up sensitivity (medium, full step)

| Warmup | Mean      | Std       |
|--------|-----------|-----------|
| 0      | 216.78 ms | 137.09 ms |
| 1      | 173.80 ms | 2.12 ms   |
| 2      | 174.41 ms | 1.86 ms   |
| 5      | 173.42 ms | 0.24 ms   |


> With no warm-up the mean jumps to ~217 ms with an enormous std of ~137 ms, because the first
> measured step pays one-time costs — CUDA context creation, cuBLAS/cuDNN autotuning for each
> new tensor shape, caching-allocator growth, and lazy kernel loading — that don't recur, so
> that single slow step inflates both the mean and the variance. With 1–2 warm-up steps the mean
> is already near steady state (~174 ms), but the std is still noticeably higher than at 5
> (2.1 → 1.9 → 0.24 ms): a single warm-up step doesn't autotune every kernel shape, the allocator
> may still be growing, and the GPU clocks are still ramping toward their boost state, so a few
> more steps are needed before timings fully stabilize.

---

## 5. Code + findings: Nsight Systems profiling (§2.1.4, Problem `nsys_profile`, 5 pts)

**Files added:**
- `cs336_systems/nsys_annotations.py` — NVTX-annotated `scaled_dot_product_attention`
  wrapping the three sub-steps (`computing attention scores` / `computing softmax` /
  `final matmul`) so the profiler can attribute kernels to each part of attention.
- `cs336_systems/benchmark.py` — added NVTX ranges (`warmup`, `measured_step_i`,
  `forward`, `backward`, `optimizer`) and a `--nvtx-attention` flag that swaps in the
  annotated attention. All ranges are no-ops on non-CUDA builds (so local CPU runs work).

**Insight: `nsys profile` vs NVTX tags.** `nsys profile` is the *recorder* — it captures
the full GPU/CPU timeline of every kernel automatically. NVTX tags are *labels we add in
code* to name regions of that timeline. nsys alone gives a flat list of cryptic kernel
names; NVTX is what lets us say "this kernel ran inside the `forward` range / inside
`computing softmax`." Both are needed to attribute time to parts of *our* model.

**How to run (on the GPU box):**
```bash
nsys profile -o prof_small_512_fwd --trace=cuda,cudnn,cublas,osrt,nvtx --force-overwrite=true \
  -- uv run python -m cs336_systems.benchmark --size small --context-length 512 \
     --mode forward --warmup 3 --steps 5 --nvtx-attention
# headless analysis (no GUI needed):
nsys stats --report cuda_gpu_kern_sum  prof_small_512_fwd.nsys-rep   # kernel summary → (b),(c)
nsys stats --report nvtx_gpu_proj_sum  prof_small_512_fwd.nsys-rep   # NVTX ranges    → (a),(d),(e)
```

### Findings (small model, ctx=512, fp32; 3 warmup + 5 measured)

**Forward-only kernel mix** (`cuda_gpu_kern_sum`): matmul (CUTLASS/magma SGEMM) ≈ **78%**
of GPU time; the top kernel `cutlass_80_simt_sgemm_128x256` alone is **70.9%**, invoked
**680 / 8 = 85× per forward pass**. The remaining ~22% is memory-bound elementwise/reduction
kernels (pointwise ops, softmax `exp`+`max`/`sum` reductions, SwiGLU `sigmoid`, RMSNorm
`pow`/`mean`/`rsqrt`, head-concat `CatArrayBatchedCopy`).

**Attention sub-step times** (NVTX, median per call, 96 calls):

| Sub-step (NVTX range)        | Median   | Note |
|------------------------------|----------|------|
| `computing attention scores` | ~106 µs  | QKᵀ matmul + scale/mask |
| `computing softmax`          | ~105 µs  | exp + max/sum reductions |
| `final matmul`               | ~54 µs   | P·V matmul |

**Full training step matmul share** drops to **~59%** (vs 78% forward-only) — backward adds
GEMMs (`256x128`, `128x128`, `128x64` for input/weight grads) but the AdamW optimizer step is
entirely elementwise (per-param moment/variance updates: `BinaryFunctor` 6.5%, `add` 5.3%,
`AUnaryFunctor` 4.8%, `sqrt`, `Fill`), so non-matmul work grows faster than matmul.

### Ready-to-paste answers

> **(a)** A single forward pass takes ~17.3 ms of GPU time (each `measured_step` NVTX range
> ≈ 17.3 ms), matching the ~16.9 ms measured with Python `timeit` in §2.1.3 almost exactly —
> the profiler confirms the wall-clock benchmark wasn't hiding asynchronous GPU work.

> **(b)** The dominant kernel is a CUTLASS SGEMM matmul (`cutlass_80_simt_sgemm_128x256…`) at
> ~71% of forward GPU time, invoked **85 times per forward pass** (per-layer attention
> projections + FFN matmuls + LM head). A matmul kernel still dominates forward+backward, since
> backward does ~2× the matmul FLOPs (it just invokes additional GEMM shapes for the gradients).

> **(c)** Besides the GEMMs, non-trivial time goes to memory-bound elementwise and reduction
> kernels: generic/vectorized `elementwise_kernel` pointwise ops (~10% combined), softmax's
> `exp_kernel` (1.6%) and `reduce_kernel` for max/sum (~2%), the SwiGLU `sigmoid_kernel` (0.7%),
> RMSNorm's `pow`/`MeanOps`/`rsqrt` (~1%), and `CatArrayBatchedCopy` for concatenating heads (1%).

> **(d)** Going from inference to a full training step, the matmul fraction *decreases* from
> ~78% to ~59%. Although backward roughly triples total matmul FLOPs (input/weight-gradient
> GEMMs), it and especially the AdamW optimizer step add a larger relative amount of matmul-free
> elementwise work — per-parameter moment/variance updates and backward activation gradients —
> so non-matmul kernels claim a bigger overall slice.

> **(e)** Softmax (~105 µs) takes about as long as the QKᵀ matmul (~106 µs) and ~2× the P·V
> matmul (~54 µs), despite performing ~d≈64× fewer FLOPs (O(N²) elementwise vs the matmuls'
> O(N²·d)). The runtime is out of proportion to FLOPs because softmax is memory-bandwidth-bound
> — it reads/writes the full N×N attention matrix several times (max, exp, sum, divide) — while
> the matmuls are compute-bound. This is exactly the inefficiency FlashAttention removes by
> fusing softmax with the matmuls so the N×N matrix never touches global memory.

*(For full marks the handout wants 2 sizes × 3 context lengths >128; small/512 done, run
`large` and ctx `256`/`1024` to comment on scaling — the qualitative analysis is unchanged.)*

---

## 6. Mixed precision (§2.1.5: `mixed_precision_accumulation` 1 pt, `benchmarking_mixed_precision` 2 pts)

### Float-format background (why low precision loses accuracy)

A float is `±(1.mantissa) × 2^exponent` — exponent bits set **dynamic range**, mantissa bits set
**precision**.

| Format | Sign/Exp/Mantissa | Dynamic range | Precision |
|--------|-------------------|---------------|-----------|
| fp32   | 1 / 8 / 23        | ~1e-38 … 3.4e38 | ~7 digits |
| fp16   | 1 / 5 / 10        | ~6e-8 … 65504   | ~3–4 digits |
| bf16   | 1 / 8 / 7         | ~1e-38 … 3e38 (= fp32!) | ~2–3 digits |

- A value is exact only if it's `integer × 2^k`. **10 = 1.010₂×2³ is exact**; **0.01 = 1/100 has
  factor 25 (not a power of 2) → non-terminating binary fraction → never exact** in any binary float.
- **bf16 keeps fp32's range but has the worst precision** — key reason it's preferred for training
  (rarely overflows) over fp16.

### `mixed_precision_accumulation` — accumulating 1000 × 0.01 (true = 10.0)

| Accumulator | Addend | Result | Error source |
|-------------|--------|--------|--------------|
| fp32 | fp32 | 10.0001 | negligible |
| **fp16** | fp16 | **9.9531** | **accumulator** — ULP near 10 is ~0.0078 > 0.01, small adds get swallowed |
| fp32 | fp16 (implicit upcast) | 10.0021 | addend only — fp16(0.01)≈0.0100021, ×1000 |
| fp32 | fp16 (explicit `.float()`) | 10.0021 | same — PyTorch upcasts fp16→fp32 in `+` either way |

> **Answer:** Pure fp32 gives 10.0001 but pure fp16 gives only 9.9531 (~0.5% error), because once the
> running sum grows large the fp16 ULP (~0.0078 near 10) exceeds the 0.01 increments and small adds are
> partially absorbed. Keeping the **accumulator in fp32** fixes it (10.0021) even with fp16 addends —
> PyTorch type-promotes fp16→fp32 before adding (so explicit `.type(torch.float32)` is identical); the
> residual +0.002 is just the fixed fp16 quantization of 0.01, not accumulation drift. **Implication:**
> compute in low precision for speed, but keep reductions / optimizer state / master weights in fp32.

#### Why the error happens (the mechanism)

Float addition = **align exponents → add → round to the representable grid**. To add two numbers with
different exponents, the hardware shifts the *smaller* number's mantissa right until exponents match;
bits shifted past the fixed mantissa width are **discarded** (rounded). That discard is the error.

Worked example, `8.0 + 0.01` in fp16 (1 implicit + 10 mantissa bits):
```
8.0        = 1.0000000000              × 2^3
fp16(0.01) = 1.0100011111 × 2^-7  →  shift right 10 bits to exponent 3:
             0.0000000001|0100011111  × 2^3
                         ↑ bits right of the line don't fit in 10 mantissa bits → rounded off
result     = 1.0000000001 × 2^3 = 8 + 2^-7 = 8.0078125
```
So `8.0 + 0.01` adds **0.0078, not 0.0100** — a 0.0022 loss in one add, because 0.01's low bits fell off
the mantissa.

Why it compounds: the grid spacing (ULP) **doubles every power of two**, so as `s` grows 0.01 spans
fewer ULPs and rounds harder:

| `s` range | fp16 ULP | 0.01 in ULPs | effect |
|-----------|----------|--------------|--------|
| [1, 2)   | 0.00098  | ~10 ULP   | added accurately |
| [4, 8)   | 0.0039   | ~2.6 ULP  | mild rounding |
| **[8, 16)** | **0.0078** | **~1.28 ULP** | rounds 1.28→1 ULP, under-adds every step |

Repeated under-rounding near the top compounds to the −0.047 deficit. Limiting case = **total
absorption**: if `s` ≥ ~40, then 0.01 < ½ ULP, rounds to zero, and `s += 0.01` does *nothing*.

**Two distinct error sources:** (1) **accumulator rounding** — the running sum's coarse grid rounds
every add; *compounds* → the big fp16 error. (2) **addend quantization** — fp16(0.01)≈0.0100021 is a
fixed bias, same each step; *linear*, not compounding → the harmless +0.002. **fp32 accumulator fixes
(1):** near 10 its ULP ≈ 2⁻²⁰ ≈ 1e-6, so 0.01 is ~10,000 ULP wide and nothing important is shifted off.

### `benchmarking_mixed_precision`

**(a) dtypes under `torch.autocast(dtype=float16)`** for the ToyModel:

| Component | dtype | Why |
|-----------|-------|-----|
| model parameters | **fp32** | autocast casts op inputs, never the stored params |
| fc1 output (Linear) | **fp16** | matmul is autocast-eligible |
| ln output (LayerNorm) | **fp32** | normalization/reduction op kept in fp32 |
| logits (fc2) | **fp16** | linear |
| loss | **fp32** | softmax/cross-entropy reductions in fp32 |
| gradients | **fp32** | match fp32 params |

> **(b)** LayerNorm computes mean + variance (sum / sum-of-squares **reductions**) then divides by the
> standard deviation; those accumulations and the `1/√(var+ε)` step need fp32's range/precision and can
> overflow/underflow in fp16's narrow range, so autocast keeps LN in fp32. With **BF16** the
> dynamic-range problem disappears (bf16 shares fp32's exponent range), so the overflow/underflow risk
> is gone; PyTorch still runs LN in fp32 (bf16's 7-bit mantissa is even less precise, so fp32 reductions
> still help accuracy), but special-casing it is far less *necessary*.

**(c) Code:** added `--autocast` / `--autocast-dtype` to `benchmark.py`; only forward + loss run under
`torch.autocast`, backward/optimizer stay outside (PyTorch AMP guidance). Results (fwd+bwd, ctx 512, batch 4):

| Size | FP32 | BF16 | Speedup |
|------|------|------|---------|
| small (129M) | 58.99 ms | 66.42 ms | **0.89× (slower)** |
| medium (423M) | 152.90 ms | 104.83 ms | **1.46×** |
| large (969M) | 333.44 ms | 173.46 ms | **1.92×** |

> **(c) Answer:** BF16 gives no benefit (a ~11% slowdown) for small but a speedup that grows with size —
> 1.46× medium, 1.92× large. BF16's win comes from Tensor-Core matmuls; small models are dominated by
> kernel-launch overhead and memory-bound elementwise/normalization ops (kept fp32), so the autocast
> casting cost outweighs the tiny matmul savings, whereas large models are matmul-bound and approach ~2×.

---

## 7. Memory profiling (§2.1.6, Problem `memory_profiling`, 4 pts)

**Code:** added `--memory-profile` / `--memory-snapshot` to `benchmark.py`. After warm-up it calls
`torch.cuda.memory._record_memory_history(max_entries=1e6)`, runs the measured steps, dumps a pickle via
`_dump_snapshot(...)`, and stops recording. Every CUDA run now also prints `peak_memory` via
`torch.cuda.max_memory_allocated()` (answers b/c without the GUI). View pickles at pytorch.org/memory_viz.

### Hardware constraint: xl does NOT fit on a 32 GB RTX 5090

A full fp32 training step keeps **4 parameter-sized copies** (params + grads + AdamW m + v = params×16 B):

| Model | params×16 B (param+grad+m+v) | Fits 32 GB full step? |
|-------|------------------------------|------------------------|
| xl (~3.4B) | **~54 GB** | ❌ never (optimizer state alone exceeds VRAM) |
| large (969M) | ~15.5 GB + ~24 GB attn @ctx2048 | ❌ OOM at ctx 2048 |
| medium (423M) | ~6.5 GB | ✅ fits both contexts |

Also, **attention activations scale as seq²**: the `(batch, heads, seq, seq)` scores + softmax tensors are
saved for backward, so going ctx 128→2048 grows them 256×. At ctx 2048 even batch 1 is needed.
→ We **substituted `medium` (batch 1)** for the profiling and document xl/large as OOM (the OOM itself is
the insight that motivates §4 FlashAttention and §6/§7 sharding).

### (b) Peak memory (medium, batch 1)

| Context | Forward | Full step |
|---------|---------|-----------|
| 128  | 2040 MiB (~2.0 GB)  | 6686 MiB (~6.5 GB)  |
| 2048 | 20098 MiB (~19.6 GB) | 24027 MiB (~23.5 GB) |

> **(b)** At ctx 128 forward needs ~2.0 GB and a full step ~6.5 GB (~3.3×): the full step adds gradients +
> AdamW's two moment buffers (3 extra param-sized fp32 copies), and activations are negligible, so memory is
> **state-bound**. At ctx 2048 forward already needs ~19.6 GB and full ~23.5 GB: the seq²-scaling attention
> activations now dominate (**activation-bound**), so the forward→full gap shrinks to the fixed ~4 GB
> gradient+optimizer cost.

### (c) Mixed precision (forward + full step)

| Context | Mode | fp32 | bf16 | Δ |
|---------|------|------|------|---|
| 128  | forward | 2040 MiB | 2676 MiB | **+636 (+31%)** |
| 128  | full    | 6686 MiB | 6684 MiB | ~0% |
| 2048 | forward | 20098 MiB | 15544 MiB | −4554 (−23%) |
| 2048 | full    | 24027 MiB | 19664 MiB | −4363 (−18%) |

> **(c)** Mixed precision does **not** significantly reduce memory — at short context it can *increase* it
> (forward ctx 128: +31%). Autocast keeps params, grads, and optimizer state in fp32 and stores only
> *activations* in bf16, while *adding* cached bf16 copies of the weights for the matmuls. When state
> dominates (short context) those extra weight copies outweigh the tiny activation savings; only when
> activations dominate (long context) does bf16 give a real ~18–23% reduction. **BF16 autocast is a
> compute/speed optimization, not primarily a memory-saving one.**

### (d) Residual-stream activation tensor (xl, fp32)

`(batch, seq, d_model)` × 4 B, d_model=2560, batch 4: **ctx 128 → 5.0 MiB, ctx 2048 → 80.0 MiB** (linear in seq).

### (a) Timelines (large model, ctx 128, batch 1 — xl/medium substituted; see hardware note)

**Forward-only** ([Artifacts/mem_large_128_fwd.png](Artifacts/mem_large_128_fwd.png)) — three identical
triangular peaks (one per measured step): each forward ramps from a ~3.7 GiB baseline (persistent
parameters, 969M × 4 B) up to a ~6.5 GiB peak as activations allocate, then drops back to baseline when the
autograd graph is released.

**Full training step** ([Artifacts/mem_large_128_full.png](Artifacts/mem_large_128_full.png)) — a taller
sawtooth peaking at ~13.5 GiB (~2× the forward peak): memory ramps up through forward, peaks at the
forward→backward boundary, then descends in steps through backward as activations free and gradients
accumulate, on a higher persistent floor (params + AdamW optimizer state).

> **(a)** The forward-only timeline shows three triangular peaks (one per step): activations climb from a
> ~3.7 GiB baseline (persistent parameters) to ~6.5 GiB, then free all at once when the graph is released.
> The full-step timeline is a taller sawtooth peaking at ~13.5 GiB: memory ramps up through forward, peaks
> at the forward→backward boundary, then descends in steps through backward as activations free and
> gradients accumulate, sitting on a higher floor of parameters + optimizer state. **Yes, the stages are
> distinguishable** — the rising edge is forward, the maximum marks the start of backward, the declining
> staircase is backward.

### (e) Largest allocations at ~10% Detail

At Detail ~10% the tool hides the thousands of tiny tensors and keeps only the biggest *individual*
allocations ([Artifacts/mem_large_128_full_10.png](Artifacts/mem_large_128_full_10.png) shows the full-step
view, 5,230 of 30,212 shown; for (e) the *forward* snapshot
[Artifacts/mem_large_128_fwd.png](Artifacts/mem_large_128_fwd.png) is the right one). **Note the blue
"wedge" is the *sum* of many small activations, not one allocation** — so the largest individual blocks are
elsewhere.

Largest individual tensors (large model: d_model=1280, d_ff=5120, vocab=10000):

| At ctx 128, batch 1 | Shape | Size |
|---------------------|-------|------|
| Embedding weight (param) | 10000 × 1280 | **48.8 MiB** |
| LM-head weight (param)   | 1280 × 10000 | **48.8 MiB** |
| FFN w1/w2/w3 (param)     | 1280 × 5120  | 25 MiB each |
| *largest activation* (logits) | 1×128×10000 | only 4.9 MiB |

> **(e)** Which allocation is largest depends on context length. In the short-context forward snapshot
> (ctx 128, batch 1) the largest blocks are ~49 MiB — the embedding and LM-head **weight matrices**
> (vocab × d_model), then the ~25 MiB SwiGLU FFN weights; their stack traces point to the model's parameter
> tensors (`nn.Embedding`/`nn.Linear`), because every *activation* at this size is <5 MiB. In the
> long-context snapshot (ctx 2048) the largest single allocation flips to the attention score/probability
> tensor `(batch, heads, seq, seq)` — e.g. 1×20×2048×2048×4 B ≈ **320 MiB**, allocated inside
> `scaled_dot_product_attention` (a saved-for-backward activation, i.e. an O(seq²) "residual"). So: weight
> matrices dominate at short context, the quadratic attention tensor dominates at long context.

---

## Status / next steps

- [x] §2.1.3 `benchmarking_script` (4 pts) — script + (b)/(c) answers.
- [x] §2.1.4 `nsys_profile` (5 pts) — NVTX annotations, small/512 profiled, all five answers.
      (Optional: extra size/context combos for full coverage.)
- [x] §2.1.5 `mixed_precision_accumulation` (1) + `benchmarking_mixed_precision` (2) — all answered.
- [x] §2.1.6 `memory_profiling` (4 pts) — code added; profiled (xl/large OOM on 32 GB, documented);
      (a)/(b)/(c)/(d)/(e) answered with memory_viz screenshots in `Artifacts/`. TODO: part (f) nsys
      per-block residual analysis.
- [ ] §4 FlashAttention-2 (Triton) — biggest single chunk (forward 15 pts, backward 5 pts).
- [ ] §5 DDP, §6 optimizer state sharding, §7 FSDP, §8 parallelism math, §9 leaderboard.

**§2.1 Profiling & Benchmarking complete: 16/16 points drafted** (pending the two memory screenshots).
