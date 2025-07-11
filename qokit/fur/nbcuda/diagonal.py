###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
Apply a diagonal cost term e^{-i γ H_C/2} to a *device* state-vector,
optionally stored in 8-/16-bit block-quantised form.

Public API
----------
    apply_diagonal(sv_dev, gamma, diag_dev,
                   *, quant_bits=32, block_size=256, renorm=True)
"""
from __future__ import annotations
import math
import numba.cuda as ncu
import numpy as np

from . import quant_utils_gpu as qg     # ← helper you added earlier

# ─────────────────────────────────────────────────────────────────────────────
# GPU kernel (unchanged maths)
# ─────────────────────────────────────────────────────────────────────────────
@ncu.jit
def _apply_diag_kernel(sv, gamma, diag):
    tid = ncu.grid(1)
    if tid < sv.size:
        x = 0.5 * gamma * diag[tid]
        c, s = math.cos(x), math.sin(x)
        sv[tid] *= complex(c, -s)

# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper with quant support
# ─────────────────────────────────────────────────────────────────────────────
def apply_diagonal(
    sv_dev: "ncu.devicearray",
    gamma: float,
    diag_dev: "ncu.devicearray",
    *,
    quant_bits: int = 32,
    block_size: int = 256,
    renorm: bool = True,
):
    """
    Parameters
    ----------
    sv_dev     : complex state amplitudes OR quantised ints on device
    gamma      : float
    diag_dev   : device array of real cost-Hamiltonian diagonals
    quant_bits : {32 (=no quant), 16, 8}
    """
    # 1. de-quantise if necessary (fp32 workspace)
    if quant_bits < 32:
        sv_dev = qg.dequantise_fp(sv_dev, scales_dev=None)

    # 2. launch the diagonal kernel (same launch size as before)
    threads = 256
    blocks = (sv_dev.size + threads - 1) // threads
    _apply_diag_kernel[blocks, threads](sv_dev, gamma, diag_dev)

    # 3. re-quantise if we came from int8/16
    if quant_bits < 32:
        sv_dev, _ = qg.quantise_fp(
            sv_dev,
            quant_bits=quant_bits,
            block_size=block_size,
            renorm=renorm,
        )

    return sv_dev
