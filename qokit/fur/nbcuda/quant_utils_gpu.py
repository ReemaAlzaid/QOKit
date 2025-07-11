###############################################################################
# // SPDX‑License‑Identifier: Apache‑2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Block‑wise affine quantisation helpers for the CUDA back‑end.
Matches the public API of the existing C/Python `quant_utils` modules so
high‑level code stays the same.

Public surface  ───────────────────────────────────────────────────────────────
    quantise_fp(sv_dev, quant_bits=8, block_size=256, renorm=True)
        → (q_dev, scales_dev)

    dequantise_fp(q_dev, scales_dev, block_size=256) → sv_dev

When `quant_bits == 32` the helpers bypass quantisation and simply return the
original state‑vector – convenient for CI and backwards compatibility.
"""
from __future__ import annotations

import math
from typing import Tuple, Optional

import numba.cuda as ncu
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# configuration
# ──────────────────────────────────────────────────────────────────────────────

# 256 complex numbers → 256 * 8  = 2 kB in fp32 → fits easily in shared even on
# old CC 6.0 devices.  Tweak if your GPU prefers 128‑thread launch.
_BLOCK = 256
_SUPPORTED_BITS = {8, 16, 32}  # 32 == no quantisation


# ──────────────────────────────────────────────────────────────────────────────
# kernels (private)
# ──────────────────────────────────────────────────────────────────────────────

@ncu.jit(device=True)
def _abs_val(z):
    """Return |z| as float32 from complex64 value."""
    return math.sqrt(z.real * z.real + z.imag * z.imag)


@ncu.jit
def _find_absmax_kernel(sv, tmp):  # pragma: no cover – executed on GPU
    """tmp[blockIdx.x] = max(|sv[block]|) using one warp per block."""
    sm = ncu.shared.array(shape=_BLOCK, dtype=ncu.float32)
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid

    v = 0.0
    if gid < sv.size:
        v = _abs_val(sv[gid])
    sm[tid] = v
    ncu.syncthreads()

    stride = _BLOCK // 2
    while stride:
        if tid < stride:
            sm[tid] = max(sm[tid], sm[tid + stride])
        stride //= 2
        ncu.syncthreads()

    if tid == 0:
        tmp[bid] = sm[0]


@ncu.jit
def _quantise_kernel(src, dst, scales, inv_scales, qmax):  # pragma: no cover
    """Store each complex value as int8/16 after affine scaling."""
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid
    if gid >= src.size:
        return

    s_inv = inv_scales[bid]
    if s_inv == 0.0:
        dst[gid] = 0j  # stays zero
        return

    z = src[gid]
    re_q = int(max(-qmax, min(qmax, round(z.real * s_inv))))
    im_q = int(max(-qmax, min(qmax, round(z.imag * s_inv))))
    dst[gid] = complex(re_q, im_q)


@ncu.jit
def _dequantise_kernel(src, dst, scales):  # pragma: no cover
    tid = ncu.threadIdx.x
    bid = ncu.blockIdx.x
    gid = bid * _BLOCK + tid
    if gid >= dst.size:
        return
    s = scales[bid]
    q = src[gid]
    dst[gid] = complex(q.real * s, q.imag * s)


# ──────────────────────────────────────────────────────────────────────────────
# public helpers
# ──────────────────────────────────────────────────────────────────────────────

def quantise_fp(
    sv: "ncu.device_array",  # complex64 array on device
    quant_bits: int = 8,
    block_size: int = _BLOCK,
    renorm: bool = True,
) -> Tuple["ncu.device_array", Optional["ncu.device_array"]]:
    """Return (q_dev, scales_dev) – state‑vector in int8/16 format.

    When *quant_bits* is 32 the call is a no‑op and returns (sv, None).
    """
    if quant_bits == 32:
        return sv, None
    if quant_bits not in _SUPPORTED_BITS:
        raise ValueError(f"quant_bits must be one of {_SUPPORTED_BITS}")

    n_blocks = (sv.size + block_size - 1) // block_size

    # 1. find per‑block absmax ────────────────────────────────────────────────
    absmax = ncu.device_array(n_blocks, dtype=np.float32)
    _find_absmax_kernel[(n_blocks, block_size)](sv, absmax)

    # 2. build scale & inverse scale vectors
    qmax = (1 << (quant_bits - 1)) - 1
    scales = ncu.device_array_like(absmax)
    inv_scales = ncu.device_array_like(absmax)

    ncu.elementwise(
        "float32 m, float32 qmax -> float32 s",
        "s = (m == 0.0f) ? 0.0f : m / qmax;",
    )(absmax, np.float32(qmax), scales)

    ncu.elementwise(
        "float32 s -> float32 inv",
        "inv = (s == 0.0f) ? 0.0f : 1.0f / s;",
    )(scales, inv_scales)

    # 3. allocate q‑vector (complex int16 – still 4 B per amp at int8) 
    q_dtype = np.complex64  # store re/im as 2×int32 – simpler for Numba
    q_dev = ncu.device_array(sv.shape, dtype=q_dtype)

    # 4. quantise 
    _quantise_kernel[(n_blocks, block_size)](sv, q_dev, scales, inv_scales, np.float32(qmax))

    # 5. optional global L2 renorm to keep \|ψ\| ≈ 1 after rounding
    if renorm:
        l2 = ncu.device_array(1, dtype=np.float64)
        ncu.reduce(lambda x, y: x + y, q_dev.real ** 2 + q_dev.imag ** 2, l2)
        fac = 1.0 / math.sqrt(l2.copy_to_host()[0])
        ncu.elementwise("complex64 v, float32 f -> complex64 o", "o = v * f;")(q_dev, np.float32(fac))

    return q_dev, scales


def dequantise_fp(
    q_dev: "ncu.device_array",
    scales_dev: Optional["ncu.device_array"],
    block_size: int = _BLOCK,
):
    """Return a *new* complex64 array with de‑quantised amplitudes."""
    if scales_dev is None:  # passthrough when quant_bits == 32
        return q_dev

    sv = ncu.device_array(q_dev.shape, dtype=np.complex64)
    n_blocks = (q_dev.size + block_size - 1) // block_size
    _dequantise_kernel[(n_blocks, block_size)](q_dev, sv, scales_dev)
    return sv