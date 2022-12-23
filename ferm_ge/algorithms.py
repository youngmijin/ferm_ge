from typing import Callable, DefaultDict, List, Optional, Tuple, TypeVar

import numpy as np
from numba import njit
from tqdm.std import trange

Hypothesis = TypeVar("Hypothesis")


@njit
def ge(
    alpha: float,
    b: np.ndarray,
    fraction_x: np.ndarray,
) -> float:
    """
    Calculate generalized entropy (I_alpha)
    Note that we call P_x as fraction_x here.
    (reference: definition 4.1 at section 4.1 in the paper)
    """

    assert b.ndim == 1 and fraction_x.ndim == 1
    assert len(b) == len(fraction_x)

    x = b / np.dot(b, fraction_x)

    if alpha == 0:
        return float(np.dot(-np.log(x), fraction_x))
    elif alpha == 1:
        return float(np.dot(x * np.log(x), fraction_x))
    else:
        return float(
            np.dot(
                (np.power(x, alpha) - 1) / (alpha * (alpha - 1)),
                fraction_x,
            )
        )


@njit
def ge_confmat(
    alpha: float, r: float, tn: float, fp: float, fn: float, tp: float
) -> float:
    """Calculate I_alpha using given confusion matrix"""

    b_stat = np.array([r, r - 1, r + 1])
    fraction_x = np.array(
        [
            (tn + tp) / (tn + fp + fn + tp),
            fn / (tn + fp + fn + tp),
            fp / (tn + fp + fn + tp),
        ]
    )

    return float(ge(alpha, b_stat, fraction_x))


@njit
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
    alpha: float,
    lambda_max: float,
    nu: float,
    r: float,
    gamma: float,
    lagrangian: Callable[[Hypothesis, float], float],
    oracle: Callable[[float], Hypothesis],
    show_progress: Optional[str] = None,
) -> Tuple[DefaultDict[Hypothesis, int], List[float]]:
    """
    Solve GE-Fairness (reference: algorithm 1 at section 5 in the paper)
    After solving, the function returns the hypothesis choice counts and the
    lambda choice history.
    """

    A_alpha: float = 1 + lambda_max * (gamma + get_Iup(alpha, r))
    B: float = gamma * lambda_max

    T: int = int(4 * (A_alpha**2) * np.log(2) / (nu**2))
    kappa: float = nu / (2 * A_alpha)

    w0: float = 1.0
    w1: float = 1.0

    lambda_0: float = 0.0
    lambda_1: float = lambda_max

    hypothesis_choice_counter: DefaultDict[Hypothesis, int] = DefaultDict(int)
    lambda_choice_history: List[float] = []

    for _ in trange(T, disable=(show_progress is None), desc=show_progress):
        lambda_t: float = (lambda_0 * w0 + lambda_1 * w1) / (w0 + w1)

        h_t = oracle(lambda_t)

        w0 = w0 * np.power(
            1.0 + kappa,
            (lagrangian(h_t, lambda_0) + B) / A_alpha,
        )

        w1 = w1 * np.power(
            1.0 + kappa,
            (lagrangian(h_t, lambda_1) + B) / A_alpha,
        )

        hypothesis_choice_counter[h_t] += 1
        lambda_choice_history.append(lambda_t)

    return hypothesis_choice_counter, lambda_choice_history
