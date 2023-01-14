import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def calc_seo(
    total_rfp: float | None,
    total_rfn: float | None,
    group_rfp: NDArray[np.float_] | None,
    group_rfn: NDArray[np.float_] | None,
    group_size: NDArray[np.int_],
) -> tuple[float, float]:
    """Calculates scalar-valued equality odds."""

    assert not (
        (total_rfp is None) and (total_rfn is None)
    ), "total_rfp and total_rfn cannot be None at the same time"
    assert group_size.ndim == 1, "group_size must be 1-dimensional"

    division_factor = 0.0

    if total_rfp is None:
        assert (group_rfn is not None) and (group_rfn.ndim == 1)
        total_rfp_rep = np.ones(group_size.shape, dtype=np.float_)
        group_rfp = total_rfp_rep.copy()
    else:
        assert (group_rfp is not None) and (group_rfp.ndim == 1)
        division_factor += 1.0
        total_rfp_rep = np.repeat(total_rfp, group_rfp.shape[0])

    if total_rfn is None:
        assert (group_rfp is not None) and (group_rfp.ndim == 1)
        total_rfn_rep = np.ones(group_size.shape, dtype=np.float_)
        group_rfn = total_rfn_rep.copy()
    else:
        assert (group_rfn is not None) and (group_rfn.ndim == 1)
        division_factor += 1.0
        total_rfn_rep = np.repeat(total_rfn, group_rfn.shape[0])

    assert (
        total_rfp_rep.shape
        == total_rfn_rep.shape
        == group_rfp.shape
        == group_rfn.shape
        == group_size.shape
    ), "group_rfp, group_rfn, and group_size must have the same shape"

    group_ratio = group_size / np.sum(group_size)

    mseo = (
        np.dot(
            np.abs(group_rfp / total_rfp_rep - 1)
            + np.abs(group_rfn / total_rfn_rep - 1),
            group_ratio,
        )
        / division_factor
    )
    aseo = (
        np.dot(
            np.abs(total_rfp_rep - group_rfp)
            + np.abs(total_rfn_rep - group_rfn),
            group_ratio,
        )
        / division_factor
    )

    return mseo, aseo
