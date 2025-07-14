#!/usr/bin/env python3
"""
Phase-3 Full QAOA benchmark – backends × precision × mixer × ini

Usage examples:
  # single point, warm+swap, 32-bit only:
  python examples/benchmark_full.py --ini warm --mix swap

  # full quant sweep (32,16,8) on all backends, warm+swap:
  python examples/benchmark_full.py --ini warm --mix swap --sweep

  # full quant sweep AND test both inis + both mixers:
  python examples/benchmark_full.py --ini all --mix all --sweep
"""

from __future__ import annotations
import argparse, csv, time, psutil, inspect
from pathlib import Path

import numpy as np
from numba import cuda

from qokit.portfolio_optimization        import get_problem, portfolio_brute_force
from qokit.qaoa_objective_portfolio      import get_qaoa_portfolio_objective
from qokit.fur                           import SIMULATORS, get_available_simulator_names

CSV_FILE = "benchmark_full.csv"
CSV_HDR  = [
    "backend","N","K","p","ini","mixer","quant_bits",
    "time_s","mem_MiB","best_energy","approx_ratio",
]

def _detect_gpu_key() -> str:
    for k in SIMULATORS["xyring"]:
        if "cuda" in k or "gpu" in k:
            return k
    raise RuntimeError("No CUDA-capable simulator key found")

GPU_KEY = _detect_gpu_key()
CPU_KEY = "c"

def _mem_used(gpu: bool) -> float:
    if gpu:
        free, total = cuda.current_context().get_memory_info()
        return (total - free) / 2**20
    else:
        return psutil.Process().memory_info().rss / 2**20

def run_qaoa(obj, gpu: bool):
    theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, size=2*obj.p)
    t0 = time.perf_counter()
    if hasattr(obj, "minimise"):
        best = obj.minimise(theta0, method="COBYLA").fun
    else:
        best = obj(theta0)
    if gpu:
        cuda.synchronize()
    return best, time.perf_counter() - t0, _mem_used(gpu)

def build_objective(po, p, ini, mixer, backend, quant_bits):
    # 1) Base QAOA kwargs
    kw = dict(
        po_problem       = po,
        p                = p,
        ini              = ini,
        mixer            = mixer,
        simulator        = backend,
        parameterization = "theta",
    )
    # 2) detect which simulator class will be used
    fam = "xyring"   # all of swap/xy/trotter_ring live here
    sim_cls = SIMULATORS[fam].get(backend)
    if sim_cls is None:
        raise RuntimeError(f"Backend {backend!r} not available for mixer family {fam!r}")
    # 3) only pass quant_bits if the ctor accepts it
    sim_kwargs: dict[str,object] = {}
    if "quant_bits" in inspect.signature(sim_cls.__init__).parameters:
        sim_kwargs["quant_bits"] = quant_bits
    if sim_kwargs:
        kw["simulator_kwargs"] = sim_kwargs

    obj = get_qaoa_portfolio_objective(**kw)
    if not hasattr(obj, "p"):
        obj.p = p
    return obj

def bench_once(N, K, p, ini, mixer, backend, quant_bits):
    po = get_problem(N=N, K=K, q=0.5, seed=42, pre="rule")
    e_min, _, e_max, *_ = portfolio_brute_force(po)

    gpu = (backend == GPU_KEY)
    obj = build_objective(po, p, ini, mixer, backend, quant_bits)
    e, t, m = run_qaoa(obj, gpu)

    approx = (e - e_min) / (e_max - e_min)
    row = [
        backend, N, K, p, ini, mixer, quant_bits,
        f"{t:.6f}", f"{m:.1f}", f"{e:+.6f}", f"{approx:.6f}"
    ]

    write_header = not Path(CSV_FILE).exists()
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HDR)
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",       type=int, default=20)
    ap.add_argument("--K",       type=int, default=5)
    ap.add_argument("-p","--depth",   type=int, default=6)
    ap.add_argument("--ini",
                    choices=["dicke","warm","all"],
                    default="dicke",
                    help="Initialization: k-hot (dicke), greedy warm-start, or all")
    ap.add_argument("--mix",
                    choices=["swap","xy","all"],
                    default="swap",
                    help="Mixer to use: swap, xy, or all")
    ap.add_argument("--backend",
                    nargs="+",
                    default=["auto"],
                    help="Which backends: c, python, gpu, or auto")
    ap.add_argument("--sweep",
                    action="store_true",
                    help="Also sweep quant_bits ∈ {16,8} (besides 32)")
    ap.add_argument("--shuffle",
                    action="store_true",
                    help="Shuffle the full grid")
    args = ap.parse_args()

    # backends selection
    if args.backend == ["auto"]:
        backs = get_available_simulator_names("xyring")
    else:
        backs = args.backend
    # keep only the three we support
    backs = [b for b in backs if b in {CPU_KEY, "python", GPU_KEY}]
    if not backs:
        raise RuntimeError(f"No valid backends in {args.backend!r}")

    # quant bits list
    qbits_list = [32, 16, 8] if args.sweep else [32]
    # ini list
    ini_list = ["dicke","warm"] if args.ini=="all" else [args.ini]
    # mixer list
    mix_list = ["swap","xy"]         if args.mix=="all" else [args.mix]

    # build the full grid
    grid = [
        (be, qb, ini, mix)
        for be in backs
        for qb in qbits_list
        for ini in ini_list
        for mix in mix_list
    ]
    if args.shuffle:
        import random
        random.shuffle(grid)

    # run!
    for i,(be,qb,ini,mix) in enumerate(grid,1):
        print(f"[{i}/{len(grid)}] backend={be:<7}  quant_bits={qb:<2}  ini={ini:5}  mixer={mix}")
        bench_once(args.N, args.K, args.depth, ini, mix, be, qb)

    print(f"\n✅ Done → results appended to {CSV_FILE!r}")

if __name__=="__main__":
    main()
