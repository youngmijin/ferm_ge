import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def ge(
    alpha: float,
    b: NDArray[np.float_],
    fraction_x: NDArray[np.float_],
) -> float:
    """Calculates generalized entropy (I_alpha)"""

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


def ge_confmat(
    alpha: float, c: float, a: float, tn: float, fp: float, fn: float, tp: float
) -> float:
    """Calculates generalized entropy (I_alpha) using given confusion matrix"""

    b_stat = np.array([c, c - a, c + a])
    fraction_x = np.array(
        [
            (tn + tp) / (tn + fp + fn + tp),
            fn / (tn + fp + fn + tp),
            fp / (tn + fp + fn + tp),
        ]
    )

    return float(ge(alpha, b_stat, fraction_x))
