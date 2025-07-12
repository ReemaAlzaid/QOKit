###############################################################################
# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
###############################################################################
"""
Block-wise affine quantisation helpers for the CUDA back-end.
Matches the public API of the existing C/Python `quant_utils` modules.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np
import numba
import numba.cuda as ncu
from numba import float32   # ← NEW: dtype shorthands

# ─────────────────────────────────────────────────────────────────────────────
# configuration
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK = 256                          # complex numbers per CUDA block
_SUPPORTED_BITS = {8, 16, 32}         # 32 = no quantisation


# ─────────────────────────────────────────────────────────────────────────────
# kernels (private)
# ─────────────────────────────────────────────────────────────────────────────
@ncu.jit(device=True, inline=True)
def _abs_val(z):
    return math.sqrt(z.real * z.real + z.imag * z.imag)


@ncu.jit
def _find_absmax_kernel(sv, tmp):
    """tmp[blockIdx.x] = max(|sv[block]|) using one warp per block."""
    sh = ncu.shared.array(shape=_BLOCK, dtype=float32)     # FIX
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid

    v = 0.0
    if gid < sv.size:
        v = _abs_val(sv[gid])
    sh[tid] = v
    ncu.syncthreads()

    stride = _BLOCK // 2
    while stride:
        if tid < stride:
            sh[tid] = max(sh[tid], sh[tid + stride])
        stride //= 2
        ncu.syncthreads()

    if tid == 0:
        tmp[bid] = sh[0]


@ncu.jit
def _build_scales_kernel(absmax, scales, inv_scales, qmax):
    """Per-block scale and 1/scale – replacement for cuda.elementwise()."""
    i = ncu.grid(1)
    if i >= absmax.size:
        return
    m = absmax[i]
    s = 0.0 if m == 0.0 else m / qmax
    scales[i] = s
    inv_scales[i] = 0.0 if s == 0.0 else 1.0 / s



@ncu.jit
def _quantise_kernel(src, dst, scales_inv, qmax):
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid
    if gid >= src.size:
        return

    s_inv = scales_inv[bid]
    if s_inv == 0.0:
        dst[gid] = 0j
        return

    z = src[gid]
    q_max = qmax                 # local register
    re_q = int(max(-q_max, min(q_max, round(z.real * s_inv))))
    im_q = int(max(-q_max, min(q_max, round(z.imag * s_inv))))
    dst[gid] = complex(re_q, im_q)


@ncu.jit
def _dequantise_kernel(src, dst, scales):
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid
    if gid >= dst.size:
        return
    s = scales[bid]
    q = src[gid]
    dst[gid] = complex(q.real * s, q.imag * s)


# ─────────────────────────────────────────────────────────────────────────────
# public helpers
# ─────────────────────────────────────────────────────────────────────────────
def quantise_fp(
    sv: "ncu.device_array",            # complex64 state-vector on device
    quant_bits: int = 8,
    block_size: int = _BLOCK,
    renorm: bool = True,
) -> Tuple["ncu.device_array", Optional["ncu.device_array"]]:

    if quant_bits == 32:
        return sv, None
    if quant_bits not in _SUPPORTED_BITS:
        raise ValueError(f"quant_bits must be one of {_SUPPORTED_BITS}")

    n_blocks = (sv.size + block_size - 1) // block_size

    # 1. per-block abs-max
    absmax = ncu.device_array(n_blocks, dtype=np.float32)
    _find_absmax_kernel[(n_blocks, block_size)](sv, absmax)

    # 2. build scale + inverse
    qmax = np.float32((1 << (quant_bits - 1)) - 1)
    scales     = ncu.device_array_like(absmax)
    inv_scales = ncu.device_array_like(absmax)

    threads = 128
    blocks  = (n_blocks + threads - 1) // threads
    _build_scales_kernel[(blocks, threads)](absmax, scales, inv_scales, qmax)


    # 3. allocate q-vector (complex64: still 8 B per amp – good enough)
    q_dev = ncu.device_array(sv.shape, dtype=np.complex64)

    # 4. quantise
    _quantise_kernel[(n_blocks, block_size)](
        sv, q_dev, inv_scales, np.float32(qmax)
    )

    # 5. optional global L2 renorm
    if renorm:
        host = q_dev.copy_to_host()
        fac = np.float32(1.0 / np.linalg.norm(host))
        ncu.elementwise("complex64 v, float32 f -> complex64 o", "o = v * f;")(
            q_dev, fac
        )

    return q_dev, scales


def dequantise_fp(
    q_dev: "ncu.device_array",
    scales_dev: Optional["ncu.device_array"],
    block_size: int = _BLOCK,
):
    if scales_dev is None:
        return q_dev

    sv = ncu.device_array(q_dev.shape, dtype=np.complex64)
    n_blocks = (q_dev.size + block_size - 1) // block_size
    _dequantise_kernel[(n_blocks, block_size)](q_dev, sv, scales_dev)
    return sv