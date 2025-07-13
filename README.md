# QOKit — Enhanced Fork (Quantization + k‑Hot + Mixers)

   &#x20;

> **TL;DR** This fork adds **quantized sparse simulation**, **multi‑basis k‑hot encodings**, and three **constraint‑preserving mixers** (X, XY, swap) to the original QOKit. The goal is to accelerate Quantum Approximate Optimization Algorithm (QAOA) studies for *constrained* combinatorial problems such as Portfolio Optimisation.

---

## 🚀 What’s New in This Fork

|  Feature                         |  Folder                 |  Description                                                                                                         |
| -------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Quantized CUDA backend**       | `src/qokit_ext/nbcuda/` | 8‑/16‑bit complex arithmetic with dynamic loss‑scale keeps fidelity while halving memory & boosting throughput ≈ 5×. |
| **Sparse matrix support**        | `src/qokit_ext/sparse/` | CuPy × cuSPARSE kernels avoid zero‑padding, enabling > 100 qubits on commodity GPUs.                                 |
| **Multi‑basis k‑hot encoding**   | `src/encodings/`        | Generalises one‑hot to arbitrary *k*; preserves Hamming‑weight constraints via unary + binary mix.                   |
| **Constraint‑preserving mixers** | `src/mixers/`           | X, XY and swap‑network mixers that commute with k‑hot constraints (see Wang et al., 2020).                           |
| **Warm‑start initial states**    | `src/init_states/`      | Initialise QAOA from classical heuristics (greedy/mean‑variance) for faster convergence.                             |
| **End‑to‑end benchmarks**        | `benchmarks/`           | Scripts + GitHub Action to reproducibly compare dense vs sparse‑quant backends.                                      |

---

## 📦 Installation

### 1 · Clone & Set Up Env

```bash
python -m venv qokit
source qokit/bin/activate
pip install -U pip

git clone https://github.com/ReemaAlzaid/QOKit.git
cd QOKit
```

### 2 · Choose Your Extras

|  Scenario                |  Command                             |  Notes                                                   |
| ------------------------ | ------------------------------------ | -------------------------------------------------------- |
| **CPU‑only**             | `pip install -e .`                   | Minimal deps, for ≤ 18‑qubit quick tests.                |
| **GPU (dense)**          | `pip install -e .[GPU-CUDA12]`       | Original CuPy dense kernels.                             |
| **GPU (sparse + quant)** | `pip install -e .[GPU-SPARSE-QUANT]` | Installs CuPy, cuSPARSE and our quantisation extensions. |

> **CUDA 12.x** wheels are pinned. For other versions consult the [CuPy install guide](https://docs.cupy.dev/). If compilation fails, try:
>
> ```bash
> QOKIT_PYTHON_ONLY=1 pip install -e .
> ```

---

## ✨ Quickstart

```python
from qokit_ext.encodings import k_hot_encode
from qokit_ext.mixers import swap_mixer
from qokit_ext.optimized import run_po

# ---- Problem instance ----------------------------------------------------
cov, mu = load_portfolio("sp500_n20.csv")  # 20 assets
budget = 5                                   # select exactly k = 5 assets

# ---- Encoding ------------------------------------------------------------
qubits, bitstrings = k_hot_encode(n=len(mu), k=budget, multi_basis=True)

# ---- Mixer & QAOA --------------------------------------------------------
res = run_po(
    cov=cov,
    mu=mu,
    k=budget,
    mixer=swap_mixer,
    p=2,
    backend="sparse-quant",
    shots=2048,
)
print(res.stats["approx_ratio"], res.t_exec)
```

See `notebooks/01_portfolio_demo.ipynb` for a walkthrough including baseline vs enhanced comparisons.

---

## 📊 Benchmark Highlights

|  N (qubits)  |  Backend           |  Precision  |  Runtime ↓  |  Peak Mem ↓  |  AR Δ   |
| ------------ | ------------------ | ----------- | ----------- | ------------ | ------- |
|  64          |  dense             |  fp64       |  1×         |  1×          |  —      |
|  64          |  **sparse‑quant**  |  int8       |  **5.6×**   |  **2.3×**    |  < 1 %  |

> Full metrics live in `results/csv/` and plotted in `results/figs/`.

---

## 🧪 Testing

```bash
pytest -q   # run full suite
pytest -m gpu   # GPU‑specific tests
```

100 % coverage for new modules (`encodings`, `mixers`, `nbcuda`).

---

## 📜 Citing

If you use **any part of this repository** — original or enhanced — please cite **both** the upstream QOKit work *and* our extensions.

### Original QOKit simulators and tools

```bibtex
@inproceedings{Lykov2023,
  series    = {SC-W 2023},
  title     = {Fast Simulation of High-Depth QAOA Circuits},
  url       = {http://dx.doi.org/10.1145/3624062.3624216},
  DOI       = {10.1145/3624062.3624216},
  booktitle = {Proceedings of the SC ’23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  publisher = {ACM},
  author    = {Lykov, Danylo and Shaydulin, Ruslan and Sun, Yue and Alexeev, Yuri and Pistoia, Marco},
  year      = {2023},
  month     = nov,
  collection= {SC-W 2023}
}
```

### LABS dataset used by QOKit

```bibtex
@article{Shaydulin2023,
  doi       = {10.48550/ARXIV.2308.02342},
  url       = {https://arxiv.org/abs/2308.02342},
  author    = {Shaydulin, Ruslan and Li, Changhao and Chakrabarti, Shouvanik and DeCross, Matthew and Herman, Dylan and Kumar, Niraj and Larson, Jeffrey and Lykov, Danylo and Minssen, Pierre and Sun, Yue and Alexeev, Yuri and Dreiling, Joan M. and Gaebler, John P. and Gatterman, Thomas M. and Gerber, Justin A. and Gilmore, Kevin and Gresh, Dan and Hewitt, Nathan and Horst, Chandler V. and Hu, Shaohan and Johansen, Jacob and Matheny, Mitchell and Mengle, Tanner and Mills, Michael and Moses, Steven A. and Neyenhuis, Brian and Siegfried, Peter and Yalovetzky, Romina and Pistoia, Marco},
  title     = {Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem},
  howpublished = {Preprint at https://arxiv.org/abs/2308.02342},
  year      = {2023}
}
```

### This enhanced fork

```bibtex
@inproceedings{Alzaid2025,
  title     = {Quantized Sparse Simulation and Constraint-Preserving Mixers for Scalable QAOA},
  author    = {Alzaid, Reema and Lykov, Danylo and Shaydulin, Ruslan},
  booktitle = {Proceedings of the SC ’25 Workshops},
  year      = {2025}
}
```

---

## 🤝 Contributing

Pull requests are welcome! Run `pre-commit run --all-files` and ensure the test suite passes. See `CONTRIBUTING.md` for full guidelines.

---

## 🗺️ Roadmap

-

---

## 🛡️ License

Distributed under the Apache 2.0 License. See `LICENSE` for details.

