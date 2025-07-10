# _benchmark_utils.py
import time, warnings, math, itertools, subprocess, os, json
from functools import lru_cache
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------ modelling bits -------------------------
from qokit.fur import choose_simulator
from qokit.encoding.khot import encode_costs

def make_ring_terms(n, seed=42):
    rng = np.random.default_rng(seed)
    terms = []
    for i in range(n):
        j = (i + 1) % n
        terms.append((rng.uniform(-1, 1), (i, j)))
    return tuple(terms)

@lru_cache
def brute_ground_state_energy(n, terms):
    e_min = math.inf
    for bits in range(1 << n):
        z = np.array([(bits >> k) & 1 for k in range(n)], dtype=int)
        e = 0.0
        for c, (i, j) in terms:
            e += c * ((-1) ** z[i]) * ((-1) ** z[j])
        e_min = min(e_min, e)
    return e_min

def bench_case(
    backend: str,
    n: int,
    p: int,
    mixer: str,
    k_hot: int | None,
    quant_bits: int | None = 32,
    quant_mode: str = "wrapper",
):
    terms = make_ring_terms(n)
    if k_hot:
        terms = encode_costs(np.zeros(2**n), k_hot)

    sim = choose_simulator(
        backend=backend,
        n_qubits=n,
        terms=terms,
        mixer=mixer,
        k_hot=k_hot,
        quant_bits=quant_bits,
        quant_mode=quant_mode,
    )

    rng = np.random.default_rng(123)
    theta0 = rng.uniform(-np.pi, np.pi, size=2 * p)
    gammas0, betas0 = theta0[:p], theta0[p:]

    def _task():
        sv = sim.simulate_qaoa(gammas0, betas0)
        return sim.get_expectation(sv)

    t0 = time.perf_counter()
    mem_peak, energy = memory_usage(
        (lambda: _task()), retval=True, max_usage=True, interval=0.05
    )
    wall = time.perf_counter() - t0

    gap = np.nan
    if n <= 12:
        e_star = brute_ground_state_energy(n, terms)
        gap = energy - e_star

    return {
        "backend": backend,
        "N": n,
        "p": p,
        "mixer": mixer,
        "k_hot": k_hot,
        "quant_bits": quant_bits,
        "quant_mode": quant_mode,
        "wall_s": wall,
        "host_MiB": mem_peak,
        "energy": energy,
        "gap": gap,
    }
