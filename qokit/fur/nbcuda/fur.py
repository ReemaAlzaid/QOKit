###############################################################################
#  SPDX-License-Identifier: Apache-2.0
#  Copyright : JP Morgan Chase & Co.
###############################################################################
"""
Crash-proof GPU helpers for fast-unitary-rotation (**FUR**) kernels.

* Single-qubit uniform **Rx** on k-qubit tiles   (CuPy RawKernel, C++17)
* Two-qubit **XX + YY** ring / complete mixers   (Numba-CUDA)

Main safety features
--------------------
1.  Always launch the *bounds-checked* shared-memory kernel
    ``furx_kernel<>`` – never the warp variant.
2.  Extra kernel argument ``n_states`` prevents out-of-bounds stores.
3.  Works with **CuPy** *or* **Numba** device arrays (zero-copy view).
4.  Optional header search path:
       ``export CUPY_NVRTC_INCLUDE_DIRS=/usr/include:/some/other/include``
"""

from __future__ import annotations

import math
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numba.cuda

try:
    import cupy as cp                      # mandatory for RawKernels
except ImportError:                       # pragma: no cover
    cp = None
    if numba.cuda.is_available():
        warnings.warn("CuPy not found – GPU kernels disabled.", RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _as_cparray(x: Any) -> "cp.ndarray":
    """Return *x* as a CuPy view (zero-copy for Numba arrays)."""
    if isinstance(x, cp.ndarray):
        return x
    try:                                   # Numba’s device array implements
        return cp.asarray(x)               # __cuda_array_interface__
    except Exception as exc:               # pragma: no cover
        raise TypeError(
            "State-vector must be a CuPy ndarray or CUDA device array"
        ) from exc


def _include_flags() -> tuple[str, ...]:
    env = os.environ.get("CUPY_NVRTC_INCLUDE_DIRS", "")
    return tuple(f"-I{p}" for p in env.split(":") if p)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Single-qubit Rx kernels
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _get_furx_kernel(k_qubits: int, q_offset: int, state_mask: int):
    """Compile once per (k,q_offset,mask) triple and return RawKernel."""
    kernel_name = f"furx_kernel<{k_qubits},{q_offset},{state_mask}>"

    src = (Path(__file__).parent / "furx.cu").read_text("utf-8")
    src = src.encode("utf-8", "ignore").decode("ascii", "ignore")  # strip UTF-8 art

    mod = cp.RawModule(
        code=src,
        name_expressions=[kernel_name],
        options=("-std=c++17", *_include_flags()),
    )
    return mod.get_function(kernel_name)


def _launch_furx(
    d_sv: "cp.ndarray | numba.cuda.cudadrv.devicearray.DeviceNDArray",
    cos_: float,
    sin_: float,
    k_qubits: int,
    q_offset: int,
    state_mask: int,
):
    """Shared-memory FUR-Rx kernel (bounds-checked, never crashes)."""
    # convert Numba device array -> CuPy view (zero-copy)
    if not isinstance(d_sv, cp.ndarray):
        d_sv = cp.asarray(d_sv)

    if k_qubits > 11:
        raise ValueError("k_qubits must be ≤ 11 (shared-memory limit)")

    ker = _get_furx_kernel(k_qubits, q_offset, state_mask)

    threads = 1 << (k_qubits - 1)               # 2**(k-1)
    blocks  = (d_sv.size // 2 + threads - 1) // threads

    # EXTRA ARG: total # amplitudes (for the bounds checks inside the kernel)
    ker((blocks,), (threads,), (d_sv, cos_, sin_, d_sv.size))


def furx_all(d_sv: Any, theta: float, n_qubits: int):
    """
    In-place uniform **Rx(theta)** on every qubit of the GPU state-vector.
    """
    d_sv = _as_cparray(d_sv)
    mask = (d_sv.size - 1) >> 1
    c, s = math.cos(theta), -math.sin(theta)

    TILE = 6                                   # always safe (fits 64 threads)
    full = (n_qubits // TILE) * TILE
    for q in range(0, full, TILE):
        _launch_furx(d_sv, c, s, TILE, q, mask)

    rem = n_qubits - full
    if rem:
        _launch_furx(d_sv, c, s, rem, full, mask)


# backwards-compat old symbol (`furx` was a *kernel* in legacy code)
furx = _launch_furx


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Two-qubit XX + YY mixers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@numba.cuda.jit
def _furxy_kernel(x, wa, wb, q1, q2, mask1, mask2, maskm):
    n_states = len(x)
    n_groups = n_states // 4
    tid = numba.cuda.grid(1)
    if tid < n_groups:
        i0 = (tid & mask1) | ((tid & maskm) << 1) | ((tid & mask2) << 2)
        ia, ib = i0 | (1 << q1), i0 | (1 << q2)
        x[ia], x[ib] = wa * x[ia] + wb * x[ib], wb * x[ia] + wa * x[ib]


def furxy(d_sv: Any, theta: float, q1: int, q2: int):
    d_sv = _as_cparray(d_sv)
    if q1 > q2:
        q1, q2 = q2, q1
    n_states = d_sv.size
    mask1 = (1 << q1) - 1
    mask2 = (1 << (q2 - 1)) - 1
    maskm = mask1 ^ mask2
    mask2 ^= (n_states - 1) >> 2
    _furxy_kernel.forall(n_states)(
        d_sv, math.cos(theta), -1j * math.sin(theta),
        q1, q2, mask1, mask2, maskm
    )


def furxy_ring(d_sv: Any, theta: float, n_qubits: int):
    d_sv = _as_cparray(d_sv)
    for o in (0, 1):                     # even / odd pairs
        for q in range(o, n_qubits - 1, 2):
            furxy(d_sv, theta, q, q + 1)
    furxy(d_sv, theta, 0, n_qubits - 1)  # wrap-around


def furxy_complete(d_sv: Any, theta: float, n_qubits: int):
    d_sv = _as_cparray(d_sv)
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            furxy(d_sv, theta, i, j)
