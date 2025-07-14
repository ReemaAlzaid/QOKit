###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
"""
CUDA back‑end for QOKit (single‑GPU).  This file defines:

* A base‑class ``QAOAFastSimulatorGPUBase`` that adds optional **block‑wise
  affine quantisation** to the original float32 implementation.
* Three concrete simulators for the FUR‑based mixers (X, XY‑ring, XY‑complete).
* **NEW:** a tiny convenience wrapper ``simulate_qaoa(...)`` so user code can
  simply import one symbol:

    >>> from QOKit.fur.nbcuda.qaoa_simulator import simulate_qaoa

  without manually instantiating the classes.  The wrapper delegates to the
  correct concrete simulator and keeps the API identical to the C / Python
  back‑ends – plus the extra ``quant_bits`` keyword.
"""
from __future__ import annotations

from collections.abc import Sequence
import warnings
from typing import Optional, Any

import numba
import numba.cuda as ncu
import numpy as np

from ..qaoa_simulator_base import (
    QAOAFastSimulatorBase,
    ParamType,
    CostsType,
    TermsType,
)
from .qaoa_fur import (
    apply_qaoa_furx,
    apply_qaoa_furxy_complete,
    apply_qaoa_furxy_ring,
)
from ..diagonal_precomputation import precompute_gpu
from .utils import (
    norm_squared,
    initialize_uniform,
    multiply,
    sum_reduce,
    copy,
)

DeviceArray = numba.cuda.devicearray.DeviceNDArray

# ============================================================================
# Base class with optional quantisation
# ============================================================================
class QAOAFastSimulatorGPUBase(QAOAFastSimulatorBase):
    """CUDA single‑GPU simulator with optional int8 / int16 block quantisation."""

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        *,
        quant_bits: int = 32,
        block_size: int = 256,
        renorm: bool = True,
        quant_mode: str = "wrapper",  # "full" once fused kernels land
    ) -> None:
        super().__init__(n_qubits, costs, terms)

        self.quant_bits = quant_bits
        self.block_size = block_size
        self.renorm = renorm
        self.quant_mode = quant_mode.lower()

        # state-vector lives on device
        # self._sv_device: DeviceArray = ncu.device_array(self.n_states, dtype=np.complex128)   # ← use 16-byte elements
        self._sv_device: DeviceArray = ncu.device_array(self.n_states, dtype=np.complex64)

        # quant buffers (used only in wrapper‑mode)
        self._q_sv: Optional[DeviceArray] = None
        self._q_scales: Optional[DeviceArray] = None

    # ------------------------------------------------------------------ cost diagonal helpers
    def _diag_from_costs(self, costs: CostsType) -> DeviceArray:
        return ncu.to_device(costs)

    def _diag_from_terms(self, terms: TermsType) -> DeviceArray:
        out = ncu.device_array(self.n_states, dtype=np.float32)
        precompute_gpu(0, self.n_qubits, terms, out)
        return out

    # ------------------------------------------------------------------ quant helpers
    def _maybe_quantise(self) -> None:
        if self.quant_bits == 32 or self.quant_mode != "wrapper":
            return
        from .quant_utils_gpu import quantise_fp

        self._q_sv, self._q_scales = quantise_fp(
            self._sv_device,
            self.quant_bits,
            self.block_size,
            self.renorm,
        )

    def _maybe_dequantise(self) -> None:
        if (
            self.quant_bits == 32
            or self.quant_mode != "wrapper"
            or self._q_sv is None
        ):
            return
        from .quant_utils_gpu import dequantise_fp

        self._sv_device = dequantise_fp(
            self._q_sv,
            self._q_scales,
            self.block_size,
        )

    # ------------------------------------------------------------------ init / I‑O
    def _initialize(self, sv0: np.ndarray | None = None) -> None:
        if sv0 is None:
            initialize_uniform(self._sv_device)
        else:
            ncu.to_device(np.asarray(sv0, dtype=np.complex64), to=self._sv_device)
        self._q_sv = None
        self._q_scales = None

    def get_cost_diagonal(self) -> np.ndarray:
        return self._hc_diag.copy_to_host()

    # main entry called by wrapper/classes
    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> DeviceArray:
        self._initialize(sv0=sv0)
        self._apply_qaoa(list(gammas), list(betas), **kwargs)
        return self._sv_device if self.quant_mode == "full" else self._q_sv or self._sv_device

    # ------------------------------------------------------------------ measurement helpers (unchanged from original implementation)
    def get_statevector(self, result: DeviceArray, **kwargs) -> np.ndarray:
        return result.copy_to_host()

    def get_probabilities(self, result: DeviceArray, **kwargs) -> np.ndarray:
        preserve = kwargs.get("preserve_state", True)
        if preserve:
            result_copy = ncu.device_array_like(result)
            copy(result_copy, result)
            result = result_copy
        norm_squared(result)
        return result.copy_to_host().real

    def get_expectation(
        self,
        result: DeviceArray,
        costs: DeviceArray | np.ndarray | None = None,
        optimization_type: str = "min",
        **kwargs,
    ) -> float:
        costs_device = self._hc_diag if costs is None else self._diag_from_costs(costs)
        preserve = kwargs.get("preserve_state", True)
        if preserve:
            result_copy = ncu.device_array_like(result)
            copy(result_copy, result)
            result = result_copy
        norm_squared(result)
        multiply(result, costs_device)
        val = sum_reduce(result).real
        return -val if optimization_type == "max" else val

    def get_overlap(
        self,
        result: DeviceArray,
        costs: CostsType | None = None,
        indices: np.ndarray | Sequence[int] | None = None,
        optimization_type: str = "min",
        **kwargs,
    ) -> float:
        try:
            import cupy as cp
        except ImportError:
            warnings.warn("cupy not found – overlap slower.", RuntimeWarning)
            import numpy as cp

        probs = cp.asarray(self.get_probabilities(result, **kwargs))
        if indices is None:
            costs_vec = cp.asarray(self._hc_diag if costs is None else self._diag_from_costs(costs))
            target_val = costs_vec.max() if optimization_type == "max" else costs_vec.min()
            indices = costs_vec == target_val
        return probs[indices].sum().item()

    # ------------------------------------------------------------------ virtual – concrete subclasses implement
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        raise NotImplementedError

# ============================================================================
# Concrete mixers (FUR‑X, XY‑ring, XY‑complete)
# ============================================================================
class QAOAFURXSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        self._maybe_dequantise()
        apply_qaoa_furx(self._sv_device, gammas, betas, self._hc_diag, self.n_qubits)
        self._maybe_quantise()

class QAOAFURXYRingSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        self._maybe_dequantise()
        apply_qaoa_furxy_ring(
            self._sv_device, gammas, betas, self._hc_diag, self.n_qubits, n_trotters=n_trotters
        )
        self._maybe_quantise()

class QAOAFURXYCompleteSimulatorGPU(QAOAFastSimulatorGPUBase):
    """FUR‑based XY‑complete mixer U(β) = exp(-i β/4 Σ_{j<k} (X_j X_k + Y_j Y_k)).

    Parameters forwarded from the base‑class.  Only the `_apply_qaoa` body is
    mixer‑specific.
    """

    def _apply_qaoa(
        self,
        gammas: Sequence[float],
        betas: Sequence[float],
        **kwargs,
    ) -> None:
        """Run p alternating layers of

            • phase‑separator   exp(-i γ H_C / 2)
            • XY‑complete mixer defined above

        The heavy lifting lives in ``apply_qaoa_furxy_complete`` which is a
        GPU kernel stack.  We only need to handle (de)‑quantisation around the
        call when `quant_mode == "wrapper"`.
        """
        n_trotters = kwargs.get("n_trotters", 1)
        self._maybe_dequantise()  # ⇢ fp32 workspace when quantised
        apply_qaoa_furxy_complete(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters=n_trotters,
        )
        self._maybe_quantise()    # ⇢ back to int8/16 if needed

# ---------------------------------------------------------------------------
# Convenience wrapper – importable as a *function* for legacy scripts
# ---------------------------------------------------------------------------
import numpy as _np

_DEF_BACKENDS = {
    "furx": QAOAFURXSimulatorGPU,
    "furxy_ring": QAOAFURXYRingSimulatorGPU,
    "furxy_complete": QAOAFURXYCompleteSimulatorGPU,
}


def simulate_qaoa(
    circ: Any,
    *,
    depth: int,
    costs: CostsType | None = None,
    terms: TermsType | None = None,
    backend: str = "furx",  # one of _DEF_BACKENDS keys
    quant_bits: int = 32,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    **kwargs,
):
    """User‑facing helper so callers don’t need to instantiate classes.

    Parameters
    ----------
    circ        : any object with ``num_qubits`` attribute
    depth       : QAOA depth p (len(gammas) == len(betas) == depth)
    costs/terms : cost Hamiltonian as in the base API
    backend     : "furx" (X mixer), "furxy_ring", or "furxy_complete"
    quant_bits  : 32 | 16 | 8 – block‑wise quantisation precision
    gammas/ betas : optional explicit parameter lists; if omitted random values
                    in [0, 1) are used (convenient for smoke tests).
    """
    if backend not in _DEF_BACKENDS:
        raise ValueError(f"backend must be one of {_DEF_BACKENDS.keys()}")

    n = circ.num_qubits
    sim_cls = _DEF_BACKENDS[backend]
    sim = sim_cls(
        n_qubits=n,
        costs=costs,
        terms=terms,
        quant_bits=quant_bits,
        **kwargs,
    )

    if gammas is None:
        gammas = _np.random.rand(depth)
    if betas is None:
        betas = _np.random.rand(depth)

    return sim.simulate_qaoa(gammas, betas, **kwargs)

class QAOASparseSimulatorGPU(QAOAFastSimulatorGPUBase):
    """
    Toy sparse‐vector backend using CuPy’s CSR.
    Only amplitudes present in k-hot subspace stored.
    """
    def __init__(self, n_qubits, costs=None, terms=None, *,
                 quant_bits=32, **kwargs):
        super().__init__(n_qubits, costs, terms,
                         quant_bits=quant_bits, **kwargs)

    def _initialize(self, sv0: _np.ndarray | None = None):
        # override dense initialization with sparse CSR
        if sv0 is None:
            raise NotImplementedError("Sparse auto-init not supported")
        # build a CSR representation on the device
        data = cuda.to_device(_np.asarray(sv0, dtype=sv0.dtype))
        idxs = cuda.to_device(_np.nonzero(sv0)[0].astype(_np.int32))
        self._sv_device = cys.CSR_matrix((data, idxs, ...))
        self._q_sv = None
        self._q_scales = None
