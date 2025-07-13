#!/usr/bin/env python3
"""
Benchmarks the CUDA back-end twice:

  • fp32     (quant_bits = 32)
  • int8/16  (quant_bits = 8 or 16)

Prints runtime, speed-up and L2 error.

Examples
--------
python examples/bench_cuda_quant.py
python examples/bench_cuda_quant.py --n 22 --depth 6 --qb 16
"""
import argparse, time, numpy as np
from numba import cuda
from qokit.fur.nbcuda.qaoa_simulator import simulate_qaoa


# --------------------------------------------------------------------------- helpers
def _circ(n):
    return type("Circ", (), {"num_qubits": n})()


def _cost_diag(n):
    """Return a random cost diagonal of length 2**n."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(1 << n).astype(np.float32)


def _run(n, depth, qb):
    # cuda.select_device(0)
    # cuda.cudadrv.nvvm.options['arch'] = 'compute_87'
    circ  = _circ(n)
    costs = _cost_diag(n)
    t0 = time.perf_counter()
    state_dev = simulate_qaoa(
        circ,
        depth      = depth,
        costs      = costs,
        backend    = "furx",      # X mixer
        quant_bits = qb,
    )
    cuda.synchronize()
    return state_dev.copy_to_host(), time.perf_counter() - t0


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=18, help="# qubits")
    ap.add_argument("-p", "--depth", type=int, default=4, help="QAOA depth")
    ap.add_argument("--qb",   type=int, default=8, choices=[8, 16, 32],
                    help="quantised precision (8 or 16)")
    args = ap.parse_args()

    psi32, t32 = _run(args.n, args.depth, 32)
    psiQ , tQ  = _run(args.n, args.depth, args.qb)

    rel = np.linalg.norm(psi32 - psiQ) / np.linalg.norm(psi32)
    spd = t32 / tQ

    print(
        f"RESULTS (n={args.n}, depth={args.depth})\n"
        f"  fp32  : {t32:6.2f} s\n"
        f"  int{args.qb:<2} : {tQ:6.2f} s   speed-up ×{spd:4.2f}\n"
        f"  L2 error : {rel:.3e}"
    )


if __name__ == "__main__":
    main()
