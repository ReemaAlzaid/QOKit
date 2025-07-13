###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Block-wise affine quantisation helpers for the CUDA back-end.
Public API:

    quantise_fp(sv_dev, quant_bits=8, block_size=256, renorm=True)
        → (q_dev, scales_dev)

    dequantise_fp(q_dev, scales_dev, block_size=256) → sv_dev
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cupy as cp
import numba.cuda as ncu
import numpy as np

# NumPy dtypes (device_array wants these)
float32   = np.float32
complex64 = np.complex64

_BLOCK = 256
_SUPPORTED_BITS = {8, 16, 32}       # 32 = bypass

# ────────────────────────────────────────────────────────────────────────────
# kernels
# ────────────────────────────────────────────────────────────────────────────
@ncu.jit(device=True, inline=True)
def _abs_val(z: complex64) -> float32:
    return math.sqrt(z.real * z.real + z.imag * z.imag)


@ncu.jit(cache=True)
def _find_absmax_kernel(sv, tmp):
    sm = ncu.shared.array(shape=_BLOCK, dtype=float32)
    tid = ncu.threadIdx.x
    gid = ncu.blockIdx.x * _BLOCK + tid
    sm[tid] = _abs_val(sv[gid]) if gid < sv.size else 0.0
    ncu.syncthreads()

    step = _BLOCK // 2
    while step:
        if tid < step:
            sm[tid] = max(sm[tid], sm[tid + step])
        step //= 2
        ncu.syncthreads()

    if tid == 0:
        tmp[ncu.blockIdx.x] = sm[0]


@ncu.jit(cache=True)
def _quantise_kernel(src, dst, scales_inv, qmax: float32):
    gid = ncu.blockIdx.x * _BLOCK + ncu.threadIdx.x
    if gid >= src.size:
        return
    s_inv = scales_inv[ncu.blockIdx.x]
    if s_inv == 0.0:
        dst[gid] = 0j
        return
    z = src[gid]
    re_q = int(max(-qmax, min(qmax, round(z.real * s_inv))))
    im_q = int(max(-qmax, min(qmax, round(z.imag * s_inv))))
    dst[gid] = complex(re_q, im_q)


@ncu.jit(cache=True)
def _dequantise_kernel(src, dst, scales):
    gid = ncu.blockIdx.x * _BLOCK + ncu.threadIdx.x
    if gid < dst.size:
        s = scales[ncu.blockIdx.x]
        z = src[gid]
        dst[gid] = complex(z.real * s, z.imag * s)


@ncu.jit(cache=True)
def _scale_kernel(vec, factor: float32):
    gid = ncu.blockIdx.x * ncu.blockDim.x + ncu.threadIdx.x
    if gid < vec.size:
        z = vec[gid]
        vec[gid] = complex(z.real * factor, z.imag * factor)


# ────────────────────────────────────────────────────────────────────────────
# public helpers
# ────────────────────────────────────────────────────────────────────────────
def quantise_fp(
    sv: "ncu.device_array",
    quant_bits: int = 8,
    block_size: int = _BLOCK,
    renorm: bool = True,
):
    if quant_bits == 32:
        return sv, None
    if quant_bits not in _SUPPORTED_BITS:
        raise ValueError(f"quant_bits must be one of {_SUPPORTED_BITS}")

    n_blocks = (sv.size + block_size - 1) // block_size

    # 1. |·|_max per block
    absmax = ncu.device_array(n_blocks, dtype=float32)
    _find_absmax_kernel[(n_blocks, block_size)](sv, absmax)

    qmax = float32((1 << (quant_bits - 1)) - 1)

    # 2. scales (CuPy)
    absmax_cp = cp.asarray(absmax)
    scales_cp = absmax_cp / qmax
    inv_cp    = cp.where(scales_cp == 0, 0, 1 / scales_cp)

    # 3. quantise
    q_dev = ncu.device_array_like(sv)
    _quantise_kernel[(n_blocks, block_size)](sv, q_dev, inv_cp, qmax)

    # 4. optional global L2 renorm
    if renorm:
        l2 = float(cp.linalg.norm(cp.asarray(q_dev)))
        if l2 > 0:
            fac = float32(1.0 / l2)
            threads = 256
            blocks  = (q_dev.size + threads - 1) // threads
            _scale_kernel[(blocks, threads)](q_dev, fac)

    return q_dev, scales_cp


def dequantise_fp(q_dev, scales_dev, block_size: int = _BLOCK):
    if scales_dev is None:        # bypass path (quant_bits == 32)
        return q_dev
    sv = ncu.device_array(q_dev.shape, dtype=complex64)
    n_blocks = (q_dev.size + block_size - 1) // block_size
    _dequantise_kernel[(n_blocks, block_size)](q_dev, sv, scales_dev)
    return sv
