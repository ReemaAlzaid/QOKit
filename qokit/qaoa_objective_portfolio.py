from __future__ import annotations
import numpy as np
from .utils import precompute_energies, reverse_array_index_bit_order, classical_warm_start_bitstring, build_terms
from .portfolio_optimization import get_configuration_cost_kw, po_obj_func, portfolio_brute_force
from qokit.qaoa_circuit_portfolio import generate_dicke_state_fast, get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective


def get_qaoa_portfolio_objective(
    po_problem: dict,
    p: int,
    ini: str = "dicke",
    mixer: str | None = "trotter_ring",
    T: int = 1,
    precomputed_energies: np.ndarray | None = None,
    parameterization: str = "theta",
    objective: str = "expectation",
    precomputed_optimal_bitstrings: np.ndarray | None = None,
    simulator: str = "auto",
    simulator_kwargs: dict | None = None,      # ← NEW: pass quant_bits, dtype, etc.
    k_hot: str | None = None,

):
    """Return QAOA objective to be minimized

    Parameters
    ----------
    po_problem : dict
        The portfolio problem instance generated by qokit.portfolio_optimization.get_problem
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    ini: string
        the initial state
    mixer: string
        the mixer
    T: int
        the Trotter step for the mixer
    precomputed_energies : np.array
        precomputed energies to compute the QAOA expectation
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
    precomputed_optimal_bitstrings : np.ndarray
        precomputed optimal bit strings to compute the QAOA overlap
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """
    N = po_problem["N"]
    if k_hot is None:
        K = po_problem["K"]
    else:
        K = int(k_hot)  # k_hot is a string, convert to int
    if precomputed_energies is None:
        po_obj = po_obj_func(po_problem)
        precomputed_energies = reverse_array_index_bit_order(precompute_energies(po_obj, N)).real
    if simulator == "qiskit":
        parameterized_circuit = get_parameterized_qaoa_circuit(po_problem, depth=p, ini=ini, mixer=mixer, T=T)
    else:
        parameterized_circuit = None

    if ini == "dicke":
        sv0 = generate_dicke_state_fast(N, K)
    elif ini == "warm":
        # classical warm‐start biasing
        x_star = classical_warm_start_bitstring(po_problem)
        α = 0.9
        sv0 = np.ones(2**N, dtype=complex) * ((1-α)/np.sqrt(2**N - 1))
        idx = sum(b<<i for i,b in enumerate(x_star))
        sv0[idx] = α
        sv0 /= np.linalg.norm(sv0)
    else:
        raise ValueError(f"Unknown ini '{ini}', allowed ['dicke','warm']")

    if mixer in ("trotter_ring", "xy", "swap"):
        pass
    else:
        raise ValueError(f"Unknown mixer passed to get_qaoa_portfolio_objective: {mixer}, allowed ['trotter_ring', 'xy', 'swap']")

    if simulator != "qiskit":
        terms = build_terms(N, po_problem["q"], po_problem["means"], po_problem["cov"])
    else:
        terms = None

    if objective == "overlap" and precomputed_optimal_bitstrings is None:
        bf_result = portfolio_brute_force(po_problem, return_bitstring=True)
        precomputed_optimal_bitstrings = bf_result[1].reshape(1, -1)
        assert precomputed_optimal_bitstrings.shape[1] == N  # only one optimal bitstring

    def scaled_result(f):
        """Return rescaled objective function
        This is done to accomodate po_problem["scale"]

        Parameters
        ----------
        function returned from the get_qaoa_objective

        Return
        ------
        f: callable function returning overlap and negative expectation
        """

        def rescaled_f(*args):
            if objective == "expectation":
                return f(*args) / po_problem["scale"]
            elif objective == "expectation and overlap":
                res = f(*args)
                return res[0] / po_problem["scale"], res[1]
            else:
                assert objective == "overlap"
                return f(*args)

        return rescaled_f

    return scaled_result(
        get_qaoa_objective(
            N=N,
            mixer=mixer,
            n_trotters=T,
            parameterized_circuit=parameterized_circuit,
            parameterization=parameterization,
            objective=objective,
            terms=terms,
            precomputed_diagonal_hamiltonian=po_problem["scale"]*precomputed_energies,
            precomputed_optimal_bitstrings=precomputed_optimal_bitstrings,
            simulator=simulator,
            simulator_kw=simulator_kwargs or {},
            initial_state=sv0,
            k_hot=k_hot
        )
    )