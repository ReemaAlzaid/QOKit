###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
High-level NumPy implementations of the three FUR-based QAOA mixers:

    • X-mixer              –  furx_all
    • XY ring-mixer        –  furxy_ring
    • XY complete-mixer    –  furxy_complete

The routines below act **in-place** on a host NumPy state-vector.  
They now accept an extra keyword, `quant_bits`, purely for API symmetry with the
CUDA back-end; the value is ignored because NumPy arrays stay in fp64.

Example
-------
>>> apply_qaoa_furx(psi, gammas, betas, hc_diag, n_qubits, quant_bits=16)

Passing `quant_bits` lets the same call-site work for either the GPU or CPU
implementation without branching.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np

# Low-level building blocks implemented in Cython / Numba-CUDA
from .fur import furx_all, furxy_ring, furxy_complete

# --------------------------------------------------------------------------- #
# Helper – apply phase-separator in one vectorised NumPy call
# --------------------------------------------------------------------------- #
def _apply_phase_separator(sv: np.ndarray, gamma: float, hc_diag: np.ndarray) -> None:
    """
    Multiply state-vector by the diagonal phase operator
        exp(-i * γ * H_C / 2)
    where `hc_diag` already contains the eigenvalues of H_C.

    This operates in-place on the NumPy array `sv`.
    """
    sv *= np.exp(-0.5j * gamma * hc_diag, dtype=sv.dtype, casting="unsafe")

# --------------------------------------------------------------------------- #
# Public mixer wrappers
# --------------------------------------------------------------------------- #
def apply_qaoa_furx(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    *,
    quant_bits: int = 32,   # accepted but ignored – for call-site parity
) -> None:
    """
    QAOA layer sequence using the **X mixer**:

        U(β) = ∏_j  exp(-i β X_j / 2)

    Parameters
    ----------
    sv        : NumPy complex array, length 2**n_qubits (modified in-place)
    gammas    : sequence of γ values (phase-separator angles)
    betas     : sequence of β values (mixer angles)
    hc_diag   : NumPy array of the diagonal cost Hamiltonian ⟨z|H_C|z⟩
    n_qubits  : total number of qubits represented by `sv`
    quant_bits: ignored here, present for API consistency with GPU path
    """
    for gamma, beta in zip(gammas, betas):
        _apply_phase_separator(sv, gamma, hc_diag)
        furx_all(sv, beta, n_qubits)           # mixer

# --------------------------------------------------------------------------- #
def apply_qaoa_furxy_ring(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int = 1,
    *,
    quant_bits: int = 32,
) -> None:
    """
    QAOA with the **XY-ring mixer**:

        U(β) = ∏_j  exp[-i β ( X_j X_{j+1} + Y_j Y_{j+1} ) / 4 ]

    Implements optional Trotterisation with `n_trotters` steps.

    Notes
    -----
    * Operates entirely on host NumPy arrays (no GPU).
    * `quant_bits` again is ignored but retained in the signature.
    """
    for gamma, beta in zip(gammas, betas):
        _apply_phase_separator(sv, gamma, hc_diag)
        step = beta / n_trotters
        for _ in range(n_trotters):
            furxy_ring(sv, step, n_qubits)

# --------------------------------------------------------------------------- #
def apply_qaoa_furxy_complete(
    sv: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    hc_diag: np.ndarray,
    n_qubits: int,
    n_trotters: int = 1,
    *,
    quant_bits: int = 32,
) -> None:
    """
    QAOA with the **XY-complete mixer**:

        U(β) = ∏_{j<k}  exp[-i β ( X_j X_k + Y_j Y_k ) / 4 ]

    This is the dense, all-to-all version.  Like the ring variant,
    it supports first-order Trotterisation.
    """
    for gamma, beta in zip(gammas, betas):
        _apply_phase_separator(sv, gamma, hc_diag)
        step = beta / n_trotters
        for _ in range(n_trotters):
            furxy_complete(sv, step, n_qubits)
