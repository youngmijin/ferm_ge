import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.warn(
    """
    NOTE: THIS PYTHON IMPLEMENTATION IS NOT USED FOR EVALUATION.
    We leave this script for those who are more familiar with Python than C.
    But there is no algorithmic difference between the two implementations.
    """,
    RuntimeWarning,
)


@dataclass
class GEFairResult:
    T: int
    D_bar: Dict[float, int]
    lambda_bar: np.ndarray
    hypothesis_history: Optional[np.ndarray]
    I_alpha_history: Optional[np.ndarray]
    err_history: Optional[np.ndarray]


def calc_L(
    ge_err_cache: Dict[float, Tuple[float, float]],
    threshold: float,
    lambda_: float,
    gamma: float,
) -> float:
    """
    Find Lagrangian with given lambda and threshold
    """

    I_alpha, err = ge_err_cache[threshold]
    return err + lambda_ * (I_alpha - gamma)


def find_threshold(
    ge_err_cache: Dict[float, Tuple[float, float]],
    thr_cache: Dict[float, float],
    thr_candidates: List[float],
    lambda_: float,
    gamma: float,
) -> float:
    """
    Find threshold for a given lambda value (the "oracle")
    """

    if lambda_ in thr_cache:
        return thr_cache[lambda_]

    thr_of_lambda = 0.0
    min_L_value = float("inf")
    for thr in thr_candidates:
        L_value = calc_L(ge_err_cache, thr, lambda_, gamma)
        if L_value < min_L_value:
            min_L_value = L_value
            thr_of_lambda = thr

    thr_cache[lambda_] = thr_of_lambda
    return thr_of_lambda


def get_Iup(alpha: float, r: float) -> float:
    """
    Calculate Iup (reference: section 5 in the paper)
    """

    rr = (r + 1) / (r - 1)

    if alpha == 0:
        return float(np.log(rr))

    if alpha == 1:
        return float(rr * np.log(rr))

    return float((np.power(rr, alpha) - 1) / np.abs((alpha - 1) * alpha))


def solve_gefair(
    thr_candidates: List[float],
    I_alpha_cache: List[float],
    err_cache: List[float],
    alpha: float,
    lambda_max: float,
    nu: float,
    r: float,
    gamma: float,
    collect_ge_history: bool = False,
):
    """
    Solve GE-Fairness (reference: algorithm 1 at section 5 in the paper)
    Note that hypothesis is a float value in this implementation.
    """

    A_alpha: float = 1 + lambda_max * (gamma + get_Iup(alpha, r))
    B: float = gamma * lambda_max

    T: int = int(4 * (A_alpha**2) * np.log(2) / (nu**2))
    kappa: float = nu / (2 * A_alpha)

    # Implementation hack: cache the results of the oracle
    thr_cache: Dict[float, float] = {}

    # Implementation hack: change form of the I_alpha and error cache to make it
    # easier to use
    ge_err_cache: Dict[float, Tuple[float, float]] = {}
    for thr, I_alpha, err in zip(thr_candidates, I_alpha_cache, err_cache):
        ge_err_cache[thr] = (I_alpha, err)

    # Implementation hack: avoid repeated calculation of multiplicative factors
    w0_mult_cache: Dict[float, float] = {}
    w1_mult_cache: Dict[float, float] = {}
    for thr in thr_candidates:
        w0_mult_cache[thr] = np.power(
            1.0 + kappa,
            (calc_L(ge_err_cache, thr, 0.0, gamma) + B) / A_alpha,
        )
        w1_mult_cache[thr] = np.power(
            1.0 + kappa,
            (calc_L(ge_err_cache, thr, lambda_max, gamma) + B) / A_alpha,
        )

    dist = np.random.default_rng()

    w0: float = 1.0
    w1: float = 1.0

    lambda_0: float = 0.0
    lambda_1: float = lambda_max

    hypothesis_history = np.zeros((T,), dtype=float)
    lambda_bar = np.zeros((T,), dtype=float)

    for t in range(T):
        # 1. Destiny chooses lambda_t
        w0_prob_t = w0 / (w0 + w1)
        lambda_t: float = lambda_0 if dist.random() < w0_prob_t else lambda_1

        # 2. The learner chooses a hypothesis (threshold(float) in this case)
        thr_t = find_threshold(
            ge_err_cache, thr_cache, thr_candidates, lambda_t, gamma
        )

        # 3. Destiny updates the weight vector (w0, w1)
        w0 = w0 * w0_mult_cache[thr_t]
        w1 = w1 * w1_mult_cache[thr_t]

        # 4. Save the hypothesis and lambda_t
        hypothesis_history[t] = thr_t
        lambda_bar[t] = lambda_t

    # get statistics of hypothesis choices (so called D_bar)
    D_bar: Dict[float, int] = {}
    for thr in thr_candidates:
        D_bar[thr] = np.sum(hypothesis_history == thr)

    # Provide histories of hypothesis selection, I_alpha and error if requested
    # This is outside the scope of the algorithm, but is useful for debugging
    if collect_ge_history:
        I_alpha_history = np.zeros((T,), dtype=float)
        err_history = np.zeros((T,), dtype=float)
        for t in range(T):
            I_alpha_history[t], err_history[t] = ge_err_cache[
                hypothesis_history[t]
            ]
    else:
        hypothesis_history = None  # type: ignore
        I_alpha_history = None
        err_history = None

    return GEFairResult(
        T,
        D_bar,
        lambda_bar,
        hypothesis_history,
        I_alpha_history,
        err_history,
    )
