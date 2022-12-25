import numpy as np
from numba import njit


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
