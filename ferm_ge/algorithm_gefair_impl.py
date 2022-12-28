import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    T: int  # number of iterations
    D_bar_stats: Dict[float, int]  # number of times each threshold is selected
    lambda_hist: np.ndarray  # history of lambda values
    D_bar: Optional[np.ndarray]  # history of D_bar values
    I_alpha_bar: Optional[np.ndarray]  # history of time-averaged I_alpha values
    err_bar: Optional[np.ndarray]  # history of time-averaged err values


def D_bar_stats_list_to_dict(
    D_bar_stats_list: np.ndarray, thr_candidates: List[float]
) -> Dict[float, int]:
    return {thr: D_bar_stats_list[i] for i, thr in enumerate(thr_candidates)}


def find_threshold_idx(
    I_alpha_cache: List[float],
    err_cache: List[float],
    thr_candidates: List[float],
    lambda_: float,
    gamma: float,
) -> int:
    """
    Find threshold for a given lambda value (the "oracle")
    """

    thri_of_lambda = 0
    min_L_value = float("inf")
    for i in range(len(thr_candidates)):
        L_value = err_cache[i] + lambda_ * (I_alpha_cache[i] - gamma)
        if L_value < min_L_value:
            min_L_value = L_value
            thri_of_lambda = i

    return thri_of_lambda


def get_Iup(alpha: float, c: float, a: float) -> float:
    """
    Calculate Iup (reference: section 5 in the paper)
    """

    ca = (c + a) / (c - a)

    if alpha == 0:
        return float(np.log(ca))

    if alpha == 1:
        return float(ca * np.log(ca))

    return float((np.power(ca, alpha) - 1) / np.abs((alpha - 1) * alpha))


def solve_gefair_loop_traced(
    T: int,
    thr_candidates: List[float],
    I_alpha_cache: List[float],
    err_cache: List[float],
    lambda_max: float,
    w0_mult_cache: List[float],
    w1_mult_cache: List[float],
    lambda_0_thri: int,
    lambda_1_thri: int,
) -> GEFairResult:
    dist = np.random.default_rng()

    w0: float = 1.0
    w1: float = 1.0

    lambda_0: float = 0.0
    lambda_1: float = lambda_max

    D_bar_stats = np.zeros((len(thr_candidates),), dtype=int)
    lambda_hist = np.zeros((T,), dtype=float)
    D_bar = np.zeros((T,), dtype=float)
    I_alpha_bar = np.zeros((T,), dtype=float)
    err_bar = np.zeros((T,), dtype=float)

    D_hist_sum = 0.0
    I_alpha_hist_sum = 0.0
    err_hist_sum = 0.0
    for t in range(T):
        # 1. Destiny chooses lambda_t
        w0_selected = dist.random() < (w0 / (w0 + w1))
        lambda_t: float = lambda_0 if w0_selected else lambda_1

        # 2. The learner chooses a hypothesis (threshold(float) in this case)
        thri_t: int = lambda_0_thri if w0_selected else lambda_1_thri

        # 3. Destiny updates the weight vector (w0, w1)
        w0 = w0 * w0_mult_cache[thri_t]
        w1 = w1 * w1_mult_cache[thri_t]

        # 4. Save the results
        D_bar_stats[thri_t] += 1
        lambda_hist[t] = lambda_t

        D_hist_sum += thr_candidates[thri_t]
        I_alpha_hist_sum += I_alpha_cache[thri_t]
        err_hist_sum += err_cache[thri_t]

        D_bar[t] = D_hist_sum / (t + 1)
        I_alpha_bar[t] = I_alpha_hist_sum / (t + 1)
        err_bar[t] = err_hist_sum / (t + 1)

    return GEFairResult(
        T=T,
        D_bar_stats=D_bar_stats_list_to_dict(D_bar_stats, thr_candidates),
        lambda_hist=lambda_hist,
        D_bar=D_bar,
        I_alpha_bar=I_alpha_bar,
        err_bar=err_bar,
    )


def solve_gefair_loop(
    T: int,
    thr_candidates: List[float],
    lambda_max: float,
    w0_mult_cache: List[float],
    w1_mult_cache: List[float],
    lambda_0_thri: int,
    lambda_1_thri: int,
) -> GEFairResult:
    dist = np.random.default_rng()

    w0: float = 1.0
    w1: float = 1.0

    lambda_0: float = 0.0
    lambda_1: float = lambda_max

    D_bar_stats = np.zeros((len(thr_candidates),), dtype=int)
    lambda_hist = np.zeros((T,), dtype=float)

    # Implementation hack: use the "index number" of the threshold instead of
    #                      the threshold itself throughout the algorithm
    for t in range(T):
        # 1. Destiny chooses lambda_t
        w0_selected = dist.random() < (w0 / (w0 + w1))
        lambda_t: float = lambda_0 if w0_selected else lambda_1

        # 2. The learner chooses a hypothesis (threshold(float) in this case)
        thri_t: int = lambda_0_thri if w0_selected else lambda_1_thri

        # 3. Destiny updates the weight vector (w0, w1)
        w0 = w0 * w0_mult_cache[thri_t]
        w1 = w1 * w1_mult_cache[thri_t]

        # 4. Save the results
        D_bar_stats[thri_t] += 1
        lambda_hist[t] = lambda_t

    return GEFairResult(
        T=T,
        D_bar_stats=D_bar_stats_list_to_dict(D_bar_stats, thr_candidates),
        lambda_hist=lambda_hist,
        D_bar=None,
        I_alpha_bar=None,
        err_bar=None,
    )


def solve_gefair(
    thr_candidates: List[float],
    I_alpha_cache: List[float],
    err_cache: List[float],
    alpha: float,
    lambda_max: float,
    nu: float,
    c: float,
    a: float,
    gamma: float,
    collect_ge_history,
):
    """
    Solve GE-Fairness (reference: algorithm 1 at section 5 in the paper)
    Note that hypothesis is a float value in this implementation.
    """

    A_alpha: float = 1 + lambda_max * (gamma + get_Iup(alpha, c, a))
    B: float = gamma * lambda_max

    T: int = int(4 * (A_alpha**2) * np.log(2) / (nu**2))
    kappa: float = nu / (2 * A_alpha)

    # Implementation hack: avoid repeated calculation of multiplicative factors
    w0_mult_cache: List[float] = []
    w1_mult_cache: List[float] = []
    for i in range(len(thr_candidates)):
        w0_mult_cache[i] = np.power(
            1.0 + kappa,
            (err_cache[i] + B) / A_alpha,
        )
        w1_mult_cache[i] = np.power(
            1.0 + kappa,
            ((err_cache[i] + lambda_max * (I_alpha_cache[i] - gamma)) + B)
            / A_alpha,
        )

    lambda_0_thri: int = find_threshold_idx(
        I_alpha_cache, err_cache, thr_candidates, 0.0, gamma
    )
    lambda_1_thri: int = find_threshold_idx(
        I_alpha_cache, err_cache, thr_candidates, lambda_max, gamma
    )

    if collect_ge_history:
        return solve_gefair_loop_traced(
            T,
            thr_candidates,
            I_alpha_cache,
            err_cache,
            lambda_max,
            w0_mult_cache,
            w1_mult_cache,
            lambda_0_thri,
            lambda_1_thri,
        )

    return solve_gefair_loop(
        T,
        thr_candidates,
        lambda_max,
        w0_mult_cache,
        w1_mult_cache,
        lambda_0_thri,
        lambda_1_thri,
    )
