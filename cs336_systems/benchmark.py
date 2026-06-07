"""End-to-end benchmarking for the basics Transformer (Assignment 2, §2.1.3).

Times forward-only, forward+backward, and full (incl. optimizer) training steps
for a randomly-initialized BasicsTransformerLM on random data.

Examples:
    uv run python -m cs336_systems.benchmark --size small
    uv run python -m cs336_systems.benchmark --size medium --mode forward --warmup 0
    uv run python -m cs336_systems.benchmark --d-model 768 --d-ff 3072 \
        --num-layers 12 --num-heads 12 --context-length 512
"""

import argparse
import statistics
import timeit
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# NVTX is only available in CUDA builds; make ranges a no-op elsewhere (e.g. CPU/Mac).
_NVTX_ENABLED = torch.cuda.is_available()


def nvtx_range(msg: str):
    return nvtx.range(msg) if _NVTX_ENABLED else nullcontext()

# Model presets from handout Table 1 (§2.1.2): d_model, d_ff, num_layers, num_heads.
MODEL_SIZES = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

MODES = ("forward", "forward_backward", "full")


def sync(device: torch.device) -> None:
    """Block until all queued CUDA work is done; no-op on CPU/MPS."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_model(args, device: torch.device, dtype: torch.dtype) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    return model.to(device=device, dtype=dtype)


def run_step(model, optimizer, x, y, mode: str) -> None:
    """One step of the requested kind. Caller handles timing/synchronization.

    Forward is run WITH grad tracking in every mode (i.e. no torch.no_grad),
    so 'forward' and 'forward_backward' measure the same forward, and
    backward time can be derived as (forward_backward - forward).
    """
    with nvtx_range("forward"):
        logits = model(x)
    if mode == "forward":
        return

    with nvtx_range("backward"):
        loss = cross_entropy(logits, y)
        loss.backward()

    if mode == "full":
        with nvtx_range("optimizer"):
            optimizer.step()
    # Clear grads either way so backward steps don't accumulate across iterations.
    optimizer.zero_grad(set_to_none=True)


def benchmark(args) -> None:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    if args.nvtx_attention:
        # Swap in the NVTX-annotated attention so the profiler can break self-attention
        # into scores / softmax / final-matmul ranges (§2.1.4).
        from cs336_systems.nsys_annotations import annotated_scaled_dot_product_attention

        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(args, device, dtype)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Random batch of token ids + targets; reused across steps (we measure speed only).
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    # Warm-up: lets cuDNN/cuBLAS autotune, JIT/compile caches fill, allocator settle.
    # Wrapped in its own NVTX range so it can be filtered OUT of the profile.
    with nvtx_range("warmup"):
        for _ in range(args.warmup):
            run_step(model, optimizer, x, y, args.mode)
        sync(device)

    timings: list[float] = []
    for i in range(args.steps):
        with nvtx_range(f"measured_step_{i}"):
            start = timeit.default_timer()
            run_step(model, optimizer, x, y, args.mode)
            sync(device)  # ensure GPU work for THIS step is finished before stopping the clock
            timings.append(timeit.default_timer() - start)

    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"size={args.size or 'custom'} mode={args.mode} device={device.type} dtype={args.dtype} "
        f"ctx={args.context_length} batch={args.batch_size} params={n_params/1e6:.1f}M "
        f"warmup={args.warmup} steps={args.steps}"
    )
    print(f"  mean={mean*1e3:.2f} ms  std={std*1e3:.2f} ms  ({mean*1e3/args.context_length:.4f} ms/token-step)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end Transformer benchmarking (§2.1.3)")
    p.add_argument("--size", choices=sorted(MODEL_SIZES), help="model preset from Table 1")
    # Architecture (overridable; defaults match 'small' so a bare run works).
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # Benchmark knobs.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mode", choices=MODES, default="forward_backward")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--nvtx-attention", action="store_true",
                   help="swap in NVTX-annotated attention (scores/softmax/matmul) for profiling")

    args = p.parse_args()
    if args.size:  # preset fills in architecture, leaving other flags untouched
        for k, v in MODEL_SIZES[args.size].items():
            setattr(args, k.replace("-", "_"), v)
    return args


if __name__ == "__main__":
    benchmark(parse_args())
