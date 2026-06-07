"""NVTX-annotated scaled dot-product attention for Nsight Systems profiling (§2.1.4).

Mirrors cs336_basics.model.scaled_dot_product_attention exactly, but wraps the
three sub-steps (scores / softmax / final matmul) in NVTX ranges so the profiler
can attribute GPU kernels to each part of self-attention.

Install it before profiling with:
    import cs336_basics.model as m
    m.scaled_dot_product_attention = annotated_scaled_dot_product_attention
(the model's attention forward looks the name up as a module global at call time,
so reassigning it takes effect.)
"""

import math
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

# Reuse the exact helpers the reference implementation uses, so numerics are identical.
from cs336_basics.model import einsum, softmax

# NVTX is only available in CUDA builds; make ranges a no-op elsewhere (e.g. CPU/Mac).
_NVTX_ENABLED = torch.cuda.is_available()


def _range(msg: str):
    return nvtx.range(msg) if _NVTX_ENABLED else nullcontext()


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    with _range("scaled dot product attention"):
        d_k = K.shape[-1]

        with _range("computing attention scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with _range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)  # over the key dimension

        with _range("final matmul"):
            out = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return out
