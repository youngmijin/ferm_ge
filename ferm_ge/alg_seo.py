import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def calc_aseo(
    total_tn: float,
    total_fp: float,
    total_fn: float,
    total_tp: float,
    group_tn: NDArray[np.float_],
    group_fp: NDArray[np.float_],
    group_fn: NDArray[np.float_],
    group_tp: NDArray[np.float_],
) -> tuple[float, float, float, NDArray[np.float_], NDArray[np.float_]]:
    """Calculates scalar-valued equality odds with rFP, rFN values."""

    assert group_tn.ndim == group_fp.ndim == group_fn.ndim == group_tp.ndim == 1

    group_cnt = group_tn.shape[0]
    group_size = (group_tn + group_fp + group_fn + group_tp).astype(np.int_)

    assert 0 not in group_size

    total_rfp: NDArray[np.float_] = np.repeat(
        total_fp / (total_fp + total_tn + total_fn + total_tp), group_cnt
    )
    total_rfn: NDArray[np.float_] = np.repeat(
        total_fn / (total_fp + total_tn + total_fn + total_tp), group_cnt
    )
    group_rfp: NDArray[np.float_] = group_fp / (
        group_fp + group_tn + group_fn + group_tp
    )
    group_rfn: NDArray[np.float_] = group_fn / (
        group_fp + group_tn + group_fn + group_tp
    )

    assert (
        total_rfp.shape
        == total_rfn.shape
        == group_rfp.shape
        == group_rfn.shape
        == group_size.shape
    )

    group_ratio = group_size / np.sum(group_size)

    aseo_fp = np.dot(np.abs(total_rfp - group_rfp), group_ratio)
    aseo_fn = np.dot(np.abs(total_rfn - group_rfn), group_ratio)
    aseo = (aseo_fp + aseo_fn) / 2

    return (
        aseo,
        aseo_fp,
        aseo_fn,
        group_rfp,
        group_rfn,
    )
