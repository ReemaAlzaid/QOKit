import time, itertools, math, json, warnings, os
from functools import lru_cache
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from tqdm.auto import tqdm
from scipy.optimize import minimize

from qokit.portfolio_optimization import get_problem, portfolio_brute_force
from qokit.qaoa_objective_portfolio import get_qaoa_portfolio_objective

warnings.filterwarnings("ignore", category=RuntimeWarning)

CSV_PATH = "bench_po_c_big.csv"

# ─────────────────────────────────────────────────────────────────────────────
def as_numpy_po(po):
    po["means"] = np.asarray(po["means"], dtype=np.float32)
    po["cov"]   = np.asarray(po["cov"], dtype=np.float32)
    return po

@lru_cache
def build_objective(problem_json, p, mixer, quant_bits):
    po_prob = json.loads(problem_json)
    po_prob = as_numpy_po(po_prob)
    return get_qaoa_portfolio_objective(
        po_problem=po_prob,
        p=p,
        ini="dicke",
        mixer=mixer,
        T=1,
        simulator="c",
        parameterization="theta"
    )

def main():
    GRID = dict(
        N=[10, 12, 14, 16, 18, 20],
        K=[2, 3, 4, 5, 6],
        p=[1, 2, 3, 4, 5, 6],
        quant_bits=[32, 16, 8],
        mixer=["trotter_ring", "xy", "swap"]
    )

    SEED = 42
    rng = np.random.default_rng(SEED)
    keys = list(GRID.keys())

    # Load previous results if resuming
    completed = set()
    if os.path.exists(CSV_PATH):
        prev_df = pd.read_csv(CSV_PATH)
        for row in prev_df[keys].itertuples(index=False, name=None):
            completed.add(row)
        header_written = True
    else:
        header_written = False

    print("\nSweeping C-backend PO grid …")
    for vals in tqdm(list(itertools.product(*GRID.values())), desc="PO-C grid"):
        if tuple(vals) in completed:
            continue

        cfg = dict(zip(keys, vals))
        N_, K_, p_, qbits_, mixer_ = cfg.values()

        try:
            po_problem = get_problem(N=N_, K=K_, q=0.5, seed=SEED, pre="rule")
            po_problem = as_numpy_po(po_problem)
            po_prob_json = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in po_problem.items()})
            e_min, _, e_max, *_ = portfolio_brute_force(po_problem)
            obj_fun = build_objective(po_prob_json, p_, mixer_, qbits_)
        except Exception as e:
            print(f"⚠️  Failed at config: {cfg} → {e}")
            continue

        best_energy = math.inf
        times, nfevs = [], []

        for _ in range(3):
            theta0 = rng.uniform(-np.pi, np.pi, size=2 * p_)
            t0 = time.perf_counter()
            mem, result = memory_usage(
                (lambda: minimize(obj_fun, theta0, method="COBYLA")),
                retval=True, max_usage=True, interval=0.05
            )
            t1 = time.perf_counter()
            res = result
            best_energy = min(best_energy, res.fun)
            times.append(t1 - t0)
            nfevs.append(res.nfev)

        row = dict(
            **cfg,
            best_energy=best_energy,
            approx_ratio=(best_energy - e_min) / (e_max - e_min),
            avg_time_s=np.mean(times),
            std_time_s=np.std(times),
            avg_nfev=np.mean(nfevs),
            mem_peak_MiB=mem,
        )

        # Save to disk after every config
        df_row = pd.DataFrame([row])
        df_row.to_csv(CSV_PATH, mode="a", header=not header_written, index=False)
        header_written = True

    print(f"\n✅ Benchmark completed and saved incrementally to {CSV_PATH}")

if __name__ == "__main__":
    main()
