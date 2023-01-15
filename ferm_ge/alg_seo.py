import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def __calc_seo(
    total_rfp: NDArray[np.float_],
    total_rfn: NDArray[np.float_],
    group_rfp: NDArray[np.float_],
    group_rfn: NDArray[np.float_],
    group_size: NDArray[np.int_],
    division_factor: float,
) -> tuple[float, float]:

    assert (
        total_rfp.shape
        == total_rfn.shape
        == group_rfp.shape
        == group_rfn.shape
        == group_size.shape
    ), "group_rfp, group_rfn, and group_size must have the same shape"

    group_ratio = group_size / np.sum(group_size)

    mseo = np.dot(
        np.abs(group_rfp / total_rfp - 1) + np.abs(group_rfn / total_rfn - 1),
        group_ratio,
    )
    aseo = np.dot(
        np.abs(total_rfp - group_rfp) + np.abs(total_rfn - group_rfn),
        group_ratio,
    )

    return mseo / division_factor, aseo / division_factor


def calc_seo(
    total_tn: float,
    total_fp: float,
    total_fn: float,
    total_tp: float,
    group_tn: NDArray[np.float_],
    group_fp: NDArray[np.float_],
    group_fn: NDArray[np.float_],
    group_tp: NDArray[np.float_],
    exclude_rfp: bool,
    exclude_rfn: bool,
) -> tuple[float, float]:
    """Calculates scalar-valued equality odds."""

    assert not (
        exclude_rfn and exclude_rfp
    ), "exclude_rfn and exclude_rfp cannot be both True"

    division_factor = 0.0
    group_cnt = group_tn.shape[0]

    total_rfp: NDArray[np.float_] = np.ones((group_cnt,), dtype=np.float_)
    total_rfn: NDArray[np.float_] = np.ones((group_cnt,), dtype=np.float_)
    group_rfp: NDArray[np.float_] = np.ones((group_cnt,), dtype=np.float_)
    group_rfn: NDArray[np.float_] = np.ones((group_cnt,), dtype=np.float_)
    group_size = (group_tn + group_fp + group_fn + group_tp).astype(np.int_)

    assert 0 not in group_size, "group_size must not contain zero"

    if not exclude_rfp:
        if total_fp == 0:
            warnings.warn(
                "total FP is zero, so mseo will be a nan.", UserWarning
            )
        total_rfp = np.repeat(
            total_fp / (total_fp + total_tn + total_fn + total_tp), group_cnt
        )
        group_rfp = group_fp / (group_fp + group_tn + group_fn + group_tp)
        division_factor += 1.0

    if not exclude_rfn:
        if total_fn == 0:
            warnings.warn(
                "total FN is zero, so mseo will be a nan.", UserWarning
            )
        total_rfn = np.repeat(
            total_fn / (total_fp + total_tn + total_fn + total_tp), group_cnt
        )
        group_rfn = group_fn / (group_fp + group_tn + group_fn + group_tp)
        division_factor += 1.0

    mseo, aseo = __calc_seo(
        total_rfp,
        total_rfn,
        group_rfp,
        group_rfn,
        group_size,
        division_factor,
    )

    return mseo, aseo
