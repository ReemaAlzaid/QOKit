###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations

from collections.abc import Sequence
import warnings
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


from typing import Any

DeviceArray = numba.cuda.devicearray.DeviceNDArray


class QAOAFastSimulatorGPUBase(QAOAFastSimulatorBase):
    """
    Base-class for the CUDA back-end.  Adds optional block-wise affine
    quantisation identical to the C and Python simulators.

    Parameters
    ----------
    quant_bits   : 32  → no quantisation (default)
                   16  → int16   + fp16 scale per block
                   8   → int8    + fp16 scale per block
    block_size   : # amplitudes in one quantisation block (default 256)
    renorm       : L2-renormalise after quantising (keeps ∥ψ∥≈1)
    quant_mode   : "wrapper"  – de/quantise around each layer (safe)
                   "full"     – use fused kernels (after you add them)
    """

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        *,
        quant_bits: int = 32,
        block_size: int = 256,
        renorm: bool = True,
        quant_mode: str = "wrapper",
    ) -> None:
        super().__init__(n_qubits, costs, terms)

        self.quant_bits = quant_bits
        self.block_size = block_size
        self.renorm = renorm
        self.quant_mode = quant_mode.lower()

        # state-vector lives on device; default dtype = complex64
        self._sv_device: DeviceArray = ncu.device_array(
            self.n_states, dtype=np.complex64
        )

        # buffers used only when quant_mode == "wrapper"
        self._q_sv: Optional[DeviceArray] = None
        self._q_scales: Optional[DeviceArray] = None

    # ───────────────────────────────────────────────────────────────────
    # diagonal helpers (cost pre-computation stays unchanged)
    # ───────────────────────────────────────────────────────────────────
    def _diag_from_costs(self, costs: CostsType) -> DeviceArray:
        return ncu.to_device(costs)

    def _diag_from_terms(self, terms: TermsType) -> DeviceArray:
        out = ncu.device_array(self.n_states, dtype=np.float32)
        precompute_gpu(0, self.n_qubits, terms, out)
        return out

    # ───────────────────────────────────────────────────────────────────
    # quantisation helpers (no-op when quant_bits == 32 or full mode)
    # ───────────────────────────────────────────────────────────────────
    def _maybe_quantise(self) -> None:
        if self.quant_bits == 32 or self.quant_mode != "wrapper":
            return
        from .quant_utils_gpu import quantise_fp  # local import

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
            or self._q_sv is None          # ← guard: nothing to de-quant yet
        ):
            return
        from .quant_utils_gpu import dequantise_fp
        self._sv_device = dequantise_fp(
            self._q_sv,
            self._q_scales,
            self.block_size,
        )


    # ───────────────────────────────────────────────────────────────────
    # initialise / I/O
    # ───────────────────────────────────────────────────────────────────
    def _initialize(self, sv0: np.ndarray | None = None) -> None:
        if sv0 is None:
            initialize_uniform(self._sv_device)
        else:
            ncu.to_device(np.asarray(sv0, dtype=np.complex64), to=self._sv_device)

        # fresh run → no quant buffers yet
        self._q_sv = None
        self._q_scales = None

    def get_cost_diagonal(self) -> np.ndarray:  # unchanged
        return self._hc_diag.copy_to_host()

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

    def get_statevector(self, result: DeviceArray, **kwargs) -> np.ndarray:
        return result.copy_to_host()

    def get_probabilities(self, result: DeviceArray, **kwargs) -> np.ndarray:
        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = ncu.device_array_like(result_orig)
            copy(result, result_orig)
        norm_squared(result)
        return result.copy_to_host().real

    def get_expectation(
        self,
        result: DeviceArray,
        costs: DeviceArray | np.ndarray | None = None,
        optimization_type="min",
        **kwargs,
    ) -> float:
        if costs is None:
            costs = self._hc_diag
        else:
            costs = self._diag_from_costs(costs)

        preserve_state = kwargs.get("preserve_state", True)
        if preserve_state:
            result_orig = result
            result = ncu.device_array_like(result_orig)
            copy(result, result_orig)

        norm_squared(result)
        multiply(result, costs)

        if optimization_type == "max":
            return -1 * sum_reduce(result).real
        return sum_reduce(result).real

    def get_overlap(
        self,
        result: DeviceArray,
        costs: CostsType | None = None,
        indices: np.ndarray | Sequence[int] | None = None,
        optimization_type="min",
        **kwargs,
    ) -> float:
        try:
            import cupy as cp
        except ImportError:
            warnings.warn(
                "Cupy import failed; overlap calculation may be slower.", RuntimeWarning
            )
            import numpy as cp

        probs = self.get_probabilities(result, **kwargs)
        probs = cp.asarray(probs)

        if indices is None:
            costs_t = self._hc_diag if costs is None else self._diag_from_costs(costs)
            costs_t = cp.asarray(costs_t)
            val = costs_t.max() if optimization_type == "max" else costs_t.min()
            indices_sel = costs_t == val
        else:
            indices_sel = indices

        return probs[indices_sel].sum().item()

    # ───────────────────────────────────────────────────────────────────
    # subclasses fill in _apply_qaoa
    # ───────────────────────────────────────────────────────────────────
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        raise NotImplementedError


# ============================================================================
# concrete simulators – only change is the two helper calls per layer
# ============================================================================


class QAOAFURXSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        self._maybe_dequantise()
        apply_qaoa_furx(
            self._sv_device, gammas, betas, self._hc_diag, self.n_qubits
        )
        self._maybe_quantise()


class QAOAFURXYRingSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        self._maybe_dequantise()
        apply_qaoa_furxy_ring(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters=n_trotters,
        )
        self._maybe_quantise()


class QAOAFURXYCompleteSimulatorGPU(QAOAFastSimulatorGPUBase):
    def _apply_qaoa(self, gammas: Sequence[float], betas: Sequence[float], **kwargs):
        n_trotters = kwargs.get("n_trotters", 1)
        self._maybe_dequantise()
        apply_qaoa_furxy_complete(
            self._sv_device,
            gammas,
            betas,
            self._hc_diag,
            self.n_qubits,
            n_trotters=n_trotters,
        )
        self._maybe_quantise()
