import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def calc_ge(
    alpha: float,
    b: NDArray[np.float_],
    fraction_x: NDArray[np.float_],
) -> float:
    """Calculates generalized entropy (I_alpha)."""

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
def calc_ge_confmat(
    alpha: float,
    c: float,
    a: float,
    tn: float,
    fp: float,
    fn: float,
    tp: float,
) -> float:
    """Calculates generalized entropy (I_alpha) using given confusion matrix."""

    b_stat = np.array([c, c - a, c + a])
    fraction_x = np.array(
        [
            (tn + tp) / (tn + fp + fn + tp),
            fn / (tn + fp + fn + tp),
            fp / (tn + fp + fn + tp),
        ]
    )

    return float(calc_ge(alpha, b_stat, fraction_x))


@njit
def calc_ge_v(
    alpha: float,
    c: float,
    a: float,
    tn_g: NDArray[np.float_],
    fp_g: NDArray[np.float_],
    fn_g: NDArray[np.float_],
    tp_g: NDArray[np.float_],
) -> float:
    """Calculates generalized entropy (I_alpha) between groups (so-called V)."""

    assert (
        len(tn_g) == len(fp_g) == len(fn_g) == len(tp_g)
    ), "all items must have the same length"

    b_stat = np.array([c, c - a, c + a])

    mu_g = np.zeros_like(tn_g, dtype=np.float_)
    fraction_g = np.zeros_like(tn_g, dtype=np.float_)
    sum_g = 0
    for i in range(len(tn_g)):
        fraction_g[i] = tn_g[i] + fp_g[i] + fn_g[i] + tp_g[i]
        sum_g += fraction_g[i]
        fraction_x = np.array(
            [
                (tn_g[i] + tp_g[i]) / fraction_g[i],
                fn_g[i] / fraction_g[i],
                fp_g[i] / fraction_g[i],
            ]
        )
        mu_g[i] = np.dot(b_stat, fraction_x)
    fraction_g /= sum_g

    return float(calc_ge(alpha, mu_g, fraction_g))
