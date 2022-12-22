import sys
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from tqdm.std import trange

DataItem = TypeVar("DataItem")
Hypothesis = TypeVar("Hypothesis")


def ge(
    alpha: float,
    b: Dict[DataItem, float],
    X: List[DataItem],
    P_x: Dict[DataItem, float],
):
    """
    Calculate generalized entropy
    (reference: definition 4.1 at section 4.1 in the paper)
    """

    n = len(X)

    E_bx = 0.0
    for i in range(n):
        E_bx += P_x[X[i]] * b[X[i]]

    I_alpha = 0.0

    if alpha == 0:
        for i in range(n):
            x = b[X[i]] / E_bx
            I_alpha += P_x[X[i]] * -np.log(x + 1e-10)

    elif alpha == 1:
        for i in range(n):
            x = b[X[i]] / E_bx
            I_alpha += P_x[X[i]] * x * np.log(x + 1e-10)

    else:
        for i in range(n):
            x = b[X[i]] / E_bx
            I_alpha += (
                P_x[X[i]] * (np.power(x, alpha) - 1) / (alpha * (alpha - 1))
            )

    return I_alpha


def get_Iup(alpha: float, r: float):
    """
    Calculate Iup (reference: section 5 in the paper)
    """
    rr = (r + 1) / (r - 1)

    if alpha == 0:
        return np.log(rr)

    if alpha == 1:
        return rr * np.log(rr)

    return (1 / ((alpha - 1) * alpha)) * (np.power(rr, alpha) - 1)


def solve_gefair(
    alpha: float,
    lambda_max: float,
    nu: float,
    r: float,
    gamma: float,
    lagrangian: Callable[[Hypothesis, float], float],
    oracle: Callable[[float], Hypothesis],
    show_progress: bool = True,
    description: Optional[str] = "Solving GE-Fairness",
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

    iteration = (
        trange(1, T + 1, desc=description, file=sys.stdout)
        if show_progress
        else range(1, T + 1)
    )
    for t in iteration:
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
