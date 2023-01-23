import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ["calc_rfp_rfn"]


@njit
def calc_rfp_rfn(
    total_tn: float,
    total_fp: float,
    total_fn: float,
    total_tp: float,
    group_tn: NDArray[np.float_],
    group_fp: NDArray[np.float_],
    group_fn: NDArray[np.float_],
    group_tp: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_], float, float,]:
    """Calculates between-group fairness metrics such as rFP, rFN values."""

    assert group_tn.ndim == group_fp.ndim == group_fn.ndim == group_tp.ndim == 1

    total_rfp: NDArray[np.float_] = np.repeat(
        total_fp / (total_fp + total_tn), group_tn.shape[0]
    )
    total_rfn: NDArray[np.float_] = np.repeat(
        total_fn / (total_fn + total_tp), group_tn.shape[0]
    )
    group_rfp: NDArray[np.float_] = group_fp / (group_fp + group_tn)
    group_rfn: NDArray[np.float_] = group_fn / (group_fn + group_tp)

    assert (
        total_rfp.shape == total_rfn.shape == group_rfp.shape == group_rfn.shape
    )

    return (
        group_rfp,
        group_rfn,
        float(total_rfp[0]),
        float(total_rfn[0]),
    )
