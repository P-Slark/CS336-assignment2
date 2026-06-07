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

## Status / next steps

- [x] §2.1.3 `benchmarking_script` (4 pts) — script written, (b) & (c) answers drafted.
- [x] §2.1.4 `nsys_profile` (5 pts) — NVTX annotations added, small/512 profiled, all five
      answers drafted. (Still to do: extra size/context combos for full coverage.)
- [ ] §2.1.5 mixed precision (`mixed_precision_accumulation`, `benchmarking_mixed_precision`).
- [ ] §4 FlashAttention-2 (Triton) — biggest single chunk (forward 15 pts, backward 5 pts).
- [ ] §5 DDP, §6 optimizer state sharding, §7 FSDP, §8 parallelism math, §9 leaderboard.
