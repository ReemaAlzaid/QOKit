#!/usr/bin/env python
###############################################################################
#  quick-bench: QOKit CUDA 32-bit vs 8-bit quantised state-vector
# -----------------------------------------------------------------------------
#  Requirements:
#     * QOKit checkout with quant-aware kernels merged
#     * numba >= 0.59, cuda-python, numpy
#     * an NVIDIA GPU with sm_70+ and recent CUDA driver
#
#  Run:
#     python run_quant_benchmark.py            # uses default 20 qubits, depth 4
#     python run_quant_benchmark.py --n 24 --p 6 --qb 16   # try 16-bit
###############################################################################
import argparse, time, numpy as np
from numba import cuda

# -- QOKit imports -----------------------------------------------------------
from qokit.fur.python.qaoa_simulator import simulate_qaoa   # high-level driver

# --------------------------------------------------------------------------- #
def build_random_maxcut_circuit(n_qubits: int, depth: int):
    """Return a (dummy) object accepted by simulate_qaoa() plus its diag."""
    # If you already have a proper Qiskit circuit builder, import that instead.
    # Here we cheat: simulate_qaoa() only needs (circ, hc_diag) signature in
    # your repo.  We'll create a placeholder object and a random ±1 diagonal.
    class DummyCirc:  # minimal shim
        num_qubits = n_qubits
    circ = DummyCirc()
    diag = np.random.choice([-1, 1], size=1 << n_qubits).astype(np.float32)
    return circ, diag

# --------------------------------------------------------------------------- #
def run_once(n_qubits: int, depth: int, quant_bits: int):
    circ, diag = build_random_maxcut_circuit(n_qubits, depth)
    t0 = time.perf_counter()
    # simulate_qaoa returns (state_dev, tables); we only need the state
    state_dev, _ = simulate_qaoa(circ,
                                 backend="cuda",
                                 depth=depth,
                                 hc_diag=diag,
                                 quant_bits=quant_bits)
    cuda.synchronize()
    dt = time.perf_counter() - t0
    return state_dev.copy_to_host(), dt

# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="QOKit quantised benchmark")
    parser.add_argument("--n", type=int, default=20, help="number of qubits")
    parser.add_argument("--p", type=int, default=4,  help="QAOA depth (layers)")
    parser.add_argument("--qb", type=int, default=8, choices=[8, 16],
                        help="quantised precision (8 or 16)")
    args = parser.parse_args()

    print(f"▶  {args.n}-qubit Max-Cut, depth {args.p}")
    psi32, t32 = run_once(args.n, args.p, 32)
    print(f"   fp32   time = {t32:5.2f} s")

    psiQ , tQ  = run_once(args.n, args.p, args.qb)
    print(f"   int{args.qb:<2} time = {tQ:5.2f} s "
          f"(speed-up ×{t32/tQ:4.2f})")

    rel_err = np.linalg.norm(psi32 - psiQ) / np.linalg.norm(psi32)
    print(f"   relative L2 error = {rel_err:.3e}")

if __name__ == "__main__":
    main()
