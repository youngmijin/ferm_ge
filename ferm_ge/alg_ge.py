import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ["calc_ge", "calc_ge_confmat", "calc_ge_v"]


@njit
def calc_ge(
    alpha: float,
    mu: float,
    b: NDArray[np.float_],
    fraction_x: NDArray[np.float_],
) -> float:
    """Calculates generalized entropy (I_alpha)."""

    assert b.ndim == 1 and fraction_x.ndim == 1
    assert len(b) == len(fraction_x)
    assert mu != 0, "mu must not be zero"

    x = b / mu

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

    cnt = tn + fp + fn + tp
    assert cnt != 0, "cnt must not be zero"

    b_stat = np.array([c, c - a, c + a])
    fraction_x = np.array([(tn + tp) / cnt, fn / cnt, fp / cnt])

    mu = np.dot(b_stat, fraction_x)

    return float(calc_ge(alpha, mu, b_stat, fraction_x))


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
    ), "all items must have the same length (same group count)"

    tn = np.sum(tn_g)
    fp = np.sum(fp_g)
    fn = np.sum(fn_g)
    tp = np.sum(tp_g)

    b_stat = np.array([c, c - a, c + a])
    mu = np.dot(b_stat, np.array([(tn + tp), fn, fp]) / (tn + fp + fn + tp))

    item_count = 0.0
    mu_g = np.zeros((len(tn_g),), dtype=np.float_)
    fraction_g = np.zeros((len(tn_g),), dtype=np.float_)
    for i in range(len(tn_g)):
        sum_g = tn_g[i] + fp_g[i] + fn_g[i] + tp_g[i]
        assert sum_g != 0, "sum_g must not be zero"
        fraction_x = np.array(
            [
                (tn_g[i] + tp_g[i]) / sum_g,
                fn_g[i] / sum_g,
                fp_g[i] / sum_g,
            ]
        )
        mu_g[i] = np.dot(b_stat, fraction_x)
        fraction_g[i] = sum_g
        item_count += sum_g
    fraction_g /= item_count

    return float(calc_ge(alpha, mu, mu_g, fraction_g))
