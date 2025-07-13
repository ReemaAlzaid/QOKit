#!/usr/bin/env python3
"""
Phase-3 GPU benchmark – portfolio-optimisation (k-hot)
=====================================================

Measures
--------
* wall-time, peak device/host memory, best energy
* approximation ratio (vs brute-force optimum)
* optional CPU fp32 baseline

Results are **appended** to CSV_FILE so you can launch the script many times
(e.g. from GNU parallel or a Slurm array).

Typical runs
------------
# single job, swap mixer, 20 qubits, int16 back-end
python bench_po_c_gpu.py --N 20 --K 5 -p 6 --mix swap --qb 16

# 48-job sweep launched via GNU parallel
parallel -j4 --colsep ',' \
  python bench_po_c_gpu.py --N {1} --K 5 -p {2} --mix {3} --qb 8 --cpu ::: \
  18,4,swap 18,6,xy 20,4,swap 20,6,xy 22,4,swap 22,6,xy
"""
from __future__ import annotations

import argparse, csv, inspect, os, sys, time, psutil
from pathlib import Path

import numpy as np
from numba import cuda

# ────────────────────────────── QOKit imports ──────────────────────────────
from qokit.portfolio_optimization   import get_problem, portfolio_brute_force
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
from qokit.fur import SIMULATORS        # used to auto-detect valid GPU key
# ────────────────────────────────────────────────────────────────────────────

CSV_FILE = "bench_gpu_po_quant.csv"
CSV_HDR  = ["backend", "N", "K", "p", "mixer", "quant_bits",
            "time_s", "mem_MiB", "best_energy", "approx_ratio"]

# ═════════════════════════════════ helpers ═════════════════════════════════
def _detect_gpu_key() -> str:
    """Return the simulator key registered for the XY-ring CUDA backend."""
    gpu_keys = list(SIMULATORS["xyring"].keys())
    # The first CUDA-capable key is what we need.
    for k in gpu_keys:
        if "cuda" in k or "gpu" in k:
            return k
    raise RuntimeError(
        f"Could not find a CUDA simulator key in {gpu_keys}. "
        "Please check your QOKit installation."
    )

GPU_KEY  = _detect_gpu_key()   # e.g. 'cuda'  or  'cuda_xyring'
CPU_KEY  = "c"                 # name never changed across releases

def gpu_mem_used_mib() -> float:
    """Return current context’s allocated device memory (MiB)."""
    free, total = cuda.current_context().get_memory_info()
    return (total - free) / 2**20

def run_qaoa(obj, gpu: bool):
    """Optimise the objective once and measure runtime + memory."""
    theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, size=2 * obj.p)
    t0 = time.perf_counter()
    # QOKit ≥0.9 has .minimise(); older versions expect obj(theta)
    if hasattr(obj, "minimise"):
        res = obj.minimise(theta0, method="COBYLA")
        best = res.fun
    else:
        best = obj(theta0)
    cuda.synchronize() if gpu else None
    runtime = time.perf_counter() - t0
    mem = gpu_mem_used_mib() if gpu else psutil.Process().memory_info().rss / 2**20
    return best, runtime, mem

def build_objective(po, p, mixer, backend_key: str, quant_bits: int):
    """Return a configured QAOA objective for the chosen backend/precision."""
    kw = dict(
        po_problem       = po,
        p                = p,
        ini              = "dicke",          # k-hot initial state
        mixer            = mixer,
        simulator        = backend_key,      # resolved key
        parameterization = "theta",
    )
    # older QOKit didn’t have quant_bits – add only if accepted
    if "quant_bits" in inspect.signature(get_qaoa_portfolio_objective).parameters:
        kw["quant_bits"] = quant_bits
    return get_qaoa_portfolio_objective(**kw)

# ═════════════════════════════════ main ════════════════════════════════════
def bench_once(N, K, p, mixer, qb, want_cpu):
    """Run fp32-GPU, quantised-GPU (+ optional CPU) and append to CSV."""
    po = get_problem(N=N, K=K, q=0.5, seed=42, pre="rule")
    e_min, _, e_max, *_ = portfolio_brute_force(po)

    # ── GPU fp32 baseline ────────────────────────────────────────────────
    obj_fp32 = build_objective(po, p, mixer, GPU_KEY, quant_bits=32)
    e_fp32, t_fp32, mem_fp32 = run_qaoa(obj_fp32, gpu=True)

    # ── GPU quantised run ────────────────────────────────────────────────
    obj_q = build_objective(po, p, mixer, GPU_KEY, quant_bits=qb)
    e_q, t_q, mem_q = run_qaoa(obj_q, gpu=True)

    # ── optional CPU baseline ────────────────────────────────────────────
    if want_cpu:
        obj_cpu = build_objective(po, p, mixer, CPU_KEY, quant_bits=32)
        e_cpu, t_cpu, mem_cpu = run_qaoa(obj_cpu, gpu=False)

    # ── console summary ─────────────────────────────────────────────────
    print(f"\n🟢  N={N}, p={p}, mixer={mixer}")
    print(f"GPU fp32  : {t_fp32:6.2f}s  {mem_fp32:6.1f} MiB  E={e_fp32:+.4f}")
    print(f"GPU int{qb:<2}: {t_q:6.2f}s  {mem_q:6.1f} MiB  "
          f"speed-up ×{t_fp32/t_q:4.2f}  ΔE={abs(e_fp32-e_q)/(abs(e_fp32)+1e-12):.2e}")
    if want_cpu:
        print(f"CPU fp32  : {t_cpu:6.2f}s  {mem_cpu:6.1f} MiB  "
              f"overall GPU int{qb} speed-up ×{t_cpu/t_q:4.2f}")

    # ── append to CSV ───────────────────────────────────────────────────
    rows = [
        ["gpu_fp32", N, K, p, mixer, 32,
         t_fp32, mem_fp32, e_fp32, (e_fp32 - e_min)/(e_max - e_min)],

        [f"gpu_int{qb}", N, K, p, mixer, qb,
         t_q, mem_q, e_q, (e_q - e_min)/(e_max - e_min)],
    ]
    if want_cpu:
        rows.append(["cpu_fp32", N, K, p, mixer, 32,
                     t_cpu, mem_cpu, e_cpu, (e_cpu - e_min)/(e_max - e_min)])

    write_hdr = not Path(CSV_FILE).exists()
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(CSV_HDR)
        w.writerows(rows)
    print(f"✅  appended to {CSV_FILE}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("-p", "--depth", type=int, default=6)
    ap.add_argument("--mix", default="swap",
                    choices=["swap", "xy", "trotter_ring"])
    ap.add_argument("--qb", type=int, default=16, choices=[8, 16])
    ap.add_argument("--cpu", action="store_true")
    # sweep helper flags
    ap.add_argument("--sweep",  action="store_true", help="run 48-job sweep")
    ap.add_argument("--shuffle",action="store_true", help="shuffle the sweep order")
    args = ap.parse_args()

    if args.sweep:
        grid_N      = [18, 20, 22]
        grid_p      = [4, 6]
        grid_mixer  = ["swap", "xy"]
        repetitions = range(4)   # 4× reps → 48 combos

        tasks = [(N,5,p,mix,16,rep,args.cpu)
                 for rep in repetitions
                 for N   in grid_N
                 for p   in grid_p
                 for mix in grid_mixer]

        if args.shuffle:
            import random; random.shuffle(tasks)

        for i,(N,K,p,mix,qb,rep,cpu) in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]  N={N} p={p} mix={mix} int{qb} rep={rep}")
            bench_once(N,K,p,mix,qb, cpu)
    else:
        bench_once(args.N, args.K, args.depth, args.mix, args.qb, args.cpu)

# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
