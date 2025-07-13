#!/usr/bin/env python3
import argparse, importlib, inspect, time, numpy as np, sys

# --------------------------------------------------------------------------- #
def import_try(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def find_backend(n_qubits: int):
    """Return (callable simulate_qaoa, backend_tag)."""
    pkgs = ["QOKit", "qokit"]                      # capital Q *and* lower-case
    mods = [".fur.nbcuda.qaoa_simulator",          # CUDA
            ".fur.python.qaoa_simulator"]          # CPU
    for pkg in pkgs:
        for suf in mods:
            m = import_try(pkg + suf)
            if m is None:
                continue

            # 1) plain function?
            if hasattr(m, "simulate_qaoa"):
                return getattr(m, "simulate_qaoa"), f"{pkg+suf}::func"

            # 2) class with that method?
            for _, cls in inspect.getmembers(m, inspect.isclass):
                if not hasattr(cls, "simulate_qaoa"):
                    continue
                sig = inspect.signature(cls)
                kwargs = {}
                for p in sig.parameters.values():
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                        if p.default is inspect._empty:
                            # param has no default: accept only 'n_qubits'
                            if p.name == "n_qubits":
                                kwargs["n_qubits"] = n_qubits
                            else:
                                break           # cannot satisfy ctor
                    elif p.kind == p.VAR_POSITIONAL:
                        break
                else:
                    inst = cls(**kwargs)
                    return inst.simulate_qaoa, f"{pkg+suf}::{cls.__name__}"
    raise RuntimeError("No backend with usable simulate_qaoa found")

# --------------------------------------------------------------------------- #
def dummy_circ(n):
    class Circ: num_qubits = n
    return Circ()

# --------------------------------------------------------------------------- #
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--n", type=int, default=18, help="#qubits")
    pa.add_argument("--depth", "-p", type=int, default=4, help="QAOA depth")
    pa.add_argument("--qb", type=int, default=8, choices=[8, 16],
                    help="quantised bits (8/16)")
    args = pa.parse_args()

    simulate, tag = find_backend(args.n)
    print("Backend:", tag)

    circ = dummy_circ(args.n)

    # fp32
    t0 = time.perf_counter()
    psi32, *_ = simulate(circ, depth=args.depth)
    t_fp = time.perf_counter() - t0
    psi32 = np.asarray(psi32, dtype=np.complex64)

    # quantised run (only if backend supports it)
    kw = {}
    if "quant_bits" in inspect.signature(simulate).parameters:
        kw["quant_bits"] = args.qb
    else:
        print("Backend lacks 'quant_bits'; timing fp32 twice.")
        args.qb = 32

    t1 = time.perf_counter()
    psiQ, *_ = simulate(circ, depth=args.depth, **kw)
    t_q = time.perf_counter() - t1
    psiQ = np.asarray(psiQ, dtype=np.complex64)

    err = np.linalg.norm(psi32 - psiQ) / np.linalg.norm(psi32)
    print(f"\nRESULTS  (n={args.n}, depth={args.depth})")
    print(f"  fp32   : {t_fp:6.2f} s")
    print(f"  qb={args.qb:<2}: {t_q:6.2f} s  speed-up ×{t_fp/t_q:4.2f}")
    print(f"  rel-L2 error   : {err:.3e}")

if __name__ == "__main__":
    import time
    main()
