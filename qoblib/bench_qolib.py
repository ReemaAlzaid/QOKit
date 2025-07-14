#!/usr/bin/env python3
"""
Phase-3 GPU benchmark – Portfolio-Optimisation (k-hot, QOKit)

If you pass --cost_csv <path>, the script loads that *square* covariance
matrix (float CSV, e.g. from QOBLIB) and builds a PO instance with
means = 0.  Otherwise we fall back to QOKit's random generator.

Results are appended to bench_gpu_po_quant.csv so you can accumulate jobs.
"""
from __future__ import annotations
import argparse, csv, inspect, sys, time, types, psutil, math
from pathlib import Path
import numpy as np
from numba import cuda

# ───────────────────── "poor-man's" stub when cupy is absent ──────────────
try:
    import cupy as cp                        # noqa: F401
except ModuleNotFoundError:
    print("[info] CuPy not installed – creating dummy module (CPU-only run)")
    cp_stub = types.ModuleType("cupy")
    cp_stub.array  = cp_stub.asarray = lambda x, **kw: np.asarray(x, **kw)
    cp_stub.zeros  = lambda *a, **k: np.zeros(*a, **k)
    cp_stub.get_array_module = lambda x: np
    sys.modules["cupy"] = sys.modules["cupy.core"] = cp_stub
    sys.modules["cupy.cuda"] = types.ModuleType("cupy.cuda")
# ─────────────────────── QOKit imports (after stub) ───────────────────────
from qokit.portfolio_optimization   import get_problem, portfolio_brute_force
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective
try:                                     # registry moved in QOKit ≥ 0.9
    from qokit.fur import SIMULATORS
except ImportError:
    from qokit.simulator import SIMULATORS
# ───────────────────────────────────────────────────────────────────────────
CSV_FILE = "bench_gpu_po_quant.csv"
CSV_HDR  = ["backend", "N", "K", "p", "mixer", "quant_bits",
            "time_s", "mem_MiB", "best_energy", "approx_ratio"]

# ═════════════════════════════════ helpers ════════════════════════════════
def _detect_gpu_key() -> str:
    for k in SIMULATORS["xyring"]:
        if "cuda" in k or "gpu" in k:
            return k
    raise RuntimeError("No CUDA-capable simulator key found")
GPU_KEY, CPU_KEY = _detect_gpu_key(), "c"

def gpu_mem_used_mib() -> float:
    free, total = cuda.current_context().get_memory_info()
    return (total - free) / 2**20

def run_qaoa(obj, gpu: bool):
    theta0 = np.random.default_rng(0).uniform(-np.pi, np.pi, size=2 * obj.p)
    t0 = time.perf_counter()
    best = obj.minimise(theta0, method="COBYLA").fun if hasattr(obj, "minimise") else obj(theta0)
    if gpu:
        cuda.synchronize()
    runtime = time.perf_counter() - t0
    mem = gpu_mem_used_mib() if gpu else psutil.Process().memory_info().rss / 2**20
    return best, runtime, mem

def build_objective(po, p, mixer, backend_key: str, quant_bits: int | None):
    kw = dict(po_problem=po, p=p, ini="dicke", mixer=mixer,
              simulator=backend_key, parameterization="theta")
    if "quant_bits" in inspect.signature(get_qaoa_portfolio_objective).parameters:
        kw["quant_bits"] = quant_bits
    obj = get_qaoa_portfolio_objective(**kw)
    if not hasattr(obj, "p"):      # callable shim (QOKit ≥ 0.9)
        obj.p = p
    return obj

# ───────────────────── load external covariance matrix ────────────────────
def po_from_cov(csv_path: Path, K: int, dtype=np.float32):
    cov = np.loadtxt(csv_path, delimiter=",", dtype=dtype)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{csv_path} is not a square matrix")
    N = cov.shape[0]
    return dict(
        N=N,
        K=K,
        means=np.zeros(N, dtype=dtype),
        cov=cov,
        q=0.5,          # default risk aversion weight
        scale=1.0,      # arbitrary scaling to avoid KeyError
    )

# ═════════════════════════════════ main work ══════════════════════════════
def bench_once(N, K, p, mixer, qb, want_cpu, cost_csv: Path | None):
    if cost_csv is None:
        po = get_problem(N=N, K=K, q=0.5, seed=42, pre="rule")
    else:
        po = po_from_cov(cost_csv, K)
        N = po["N"]  # overwrite for pretty prints

    # get brute-force bounds (cheap)
    e_min, _, e_max, *_ = portfolio_brute_force(po)

    # GPU runs (may OOM for full-energy precompute)
    gpu_failed = False
    try:
        obj_fp32 = build_objective(po, p, mixer, GPU_KEY, quant_bits=32)
        e_fp32, t_fp32, mem_fp32 = run_qaoa(obj_fp32, gpu=True)
        obj_q = build_objective(po, p, mixer, GPU_KEY, quant_bits=qb)
        e_q, t_q, mem_q = run_qaoa(obj_q, gpu=True)
    except MemoryError:
        print(f"[warning] Skipping GPU runs: N={N} too large for energy precompute")
        gpu_failed = True
        e_fp32 = t_fp32 = mem_fp32 = e_q = t_q = mem_q = None

    # optional CPU baseline (may OOM too)
    cpu_failed = False
    if want_cpu:
        try:
            obj_cpu = build_objective(po, p, mixer, CPU_KEY, quant_bits=32)
            e_cpu, t_cpu, mem_cpu = run_qaoa(obj_cpu, gpu=False)
        except MemoryError:
            print(f"[warning] Skipping CPU baseline: N={N} too large for energy precompute")
            cpu_failed = True
            e_cpu = t_cpu = mem_cpu = None

    # ─ console ─
    print(f" 🟢  N={N}, p={p}, mixer={mixer}")
    if not gpu_failed:
        print(f"GPU fp32   : {t_fp32:6.2f}s  {mem_fp32:6.1f} MiB  E={e_fp32:+.4f}")
        print(f"GPU int{qb:<2}: {t_q:6.2f}s  {mem_q:6.1f} MiB  "
              f"×{t_fp32/t_q:4.2f} speed up  ΔE={abs(e_fp32-e_q)/(abs(e_fp32)+1e-12):.1e}")
    if want_cpu:
        if cpu_failed:
            print("⚠️  CPU baseline skipped due to MemoryError")
        else:
            print(f"CPU fp32   : {t_cpu:6.2f}s  {mem_cpu:6.1f} MiB  overall GPU×{t_cpu/t_q:4.2f}")

    # ─ CSV ─
    rows = []
    if not gpu_failed:
        rows.extend([
            ["gpu_fp32", N, K, p, mixer, 32, t_fp32, mem_fp32, e_fp32,
             (e_fp32 - e_min)/(e_max - e_min)],
            [f"gpu_int{qb}", N, K, p, mixer, qb, t_q, mem_q, e_q,
             (e_q - e_min)/(e_max - e_min)],
        ])
    if want_cpu and not cpu_failed:
        rows.append([
            "cpu_fp32", N, K, p, mixer, 32, t_cpu, mem_cpu, e_cpu,
            (e_cpu - e_min)/(e_max - e_min)
        ])

    if rows:
        write_hdr = not Path(CSV_FILE).exists()
        with open(CSV_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if write_hdr:
                w.writerow(CSV_HDR)
            w.writerows(rows)
        print(f"✅  appended to {CSV_FILE}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",  type=int, default=20,
                    help="Size if NOT using --cost_csv")
    ap.add_argument("--K",  type=int, default=5)
    ap.add_argument("-p", "--depth", type=int, default=6)
    ap.add_argument("--mix", default="swap",
                    choices=["swap", "xy", "trotter_ring"])
    ap.add_argument("--qb",  type=int, default=16, choices=[8, 16])
    ap.add_argument("--cpu", action="store_true",
                    help="also run a CPU fp32 baseline")
    ap.add_argument("--cost_csv", type=Path,
                    help="CSV with NxN covariance matrix (QOBLIB etc.)")
    ap.add_argument("--sweep",   action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        grid_N, grid_p, grid_mixer = [18, 20, 22], [4, 6], ["swap", "xy"]
        tasks = [(N, 5, p, mix, 16, rep, args.cpu, None)
                 for rep in range(4)
                 for N in grid_N
                 for p in grid_p
                 for mix in grid_mixer]
        if args.shuffle:
            import random; random.shuffle(tasks)
        for i, (N, K, p, mix, qb, rep, cpu, cmat) in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}]  N={N} p={p} mix={mix} int{qb} rep={rep}")
            bench_once(N, K, p, mix, qb, cpu, cmat)
    else:
        bench_once(args.N, args.K, args.depth, args.mix,
                   args.qb, args.cpu, args.cost_csv)

if __name__ == "__main__":
    main()
