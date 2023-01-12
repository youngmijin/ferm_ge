from typing import Generic, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

T = TypeVar("T")
V = TypeVar("V")


class Cache(Generic[T, V]):
    def __init__(self):
        self._cache: dict[frozenset[tuple[str, T]], V] = {}
        self._keylist: list[str] | None = None

    def __to_key(self, kwargs: dict[str, T]) -> frozenset[tuple[str, T]]:
        if self._keylist is None:
            self._keylist = sorted(kwargs.keys())
        assert self._keylist == sorted(kwargs.keys())
        return frozenset(zip(self._keylist, [kwargs[k] for k in self._keylist]))

    def set(self, value: V, **kwargs: T) -> None:
        self._cache[self.__to_key(kwargs)] = value

    def get(self, **kwargs: T) -> V:
        return self._cache[self.__to_key(kwargs)]


@njit
def get_time_averaged_trace(
    hypi_t: NDArray[np.intp],
    cache_by_hypi: NDArray[np.float_],
) -> NDArray[np.float_]:
    """Calculates time-averaged trace"""

    assert hypi_t.ndim == 1
    assert cache_by_hypi.ndim == 1

    trace_cumsum: NDArray[np.float_] = np.cumsum(cache_by_hypi[hypi_t])
    time = np.arange(1, len(hypi_t) + 1, dtype=np.float_)

    return trace_cumsum / time


@njit
def get_seo_trace(
    hypi_t: NDArray[np.intp],
    total_r_cache_by_hypi: tuple[NDArray[np.float_], NDArray[np.float_]],
    group_r_cache_by_hypi: tuple[NDArray[np.float_], NDArray[np.float_]],
    group_size_ratio: NDArray[np.float_],
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Calculates SEO trace"""

    assert hypi_t.ndim == 1
    assert total_r_cache_by_hypi[0].shape == total_r_cache_by_hypi[1].shape
    assert group_r_cache_by_hypi[0].shape == group_r_cache_by_hypi[1].shape
    assert total_r_cache_by_hypi[0].ndim == 1
    assert group_r_cache_by_hypi[0].ndim == 2
    assert group_r_cache_by_hypi[0].shape[1] == len(total_r_cache_by_hypi[0])
    assert group_size_ratio.ndim == 1
    assert group_size_ratio.shape[0] == group_r_cache_by_hypi[0].shape[0]

    aseo_trace = np.zeros_like(hypi_t, dtype=np.float_)
    mseo_trace = np.zeros_like(hypi_t, dtype=np.float_)
    for i in range(len(hypi_t)):
        rfp_tot = total_r_cache_by_hypi[0][hypi_t[i]]
        rfp_group = group_r_cache_by_hypi[0][:, hypi_t[i]]
        rfn_tot = total_r_cache_by_hypi[1][hypi_t[i]]
        rfn_group = group_r_cache_by_hypi[1][:, hypi_t[i]]
        aseo_trace[i] = (
            np.dot(
                (rfp_tot.repeat(rfp_group) - rfp_group).abs()
                + (rfn_tot.repeat(rfn_group) - rfn_group).abs(),
                group_size_ratio,
            ).sum()
            / 2
        )
        mseo_trace[i] = (
            np.dot(
                (rfp_tot.repeat(rfp_group) / rfp_group - 1).abs()
                + (rfn_tot.repeat(rfn_group) / rfn_group - 1).abs(),
                group_size_ratio,
            ).sum()
            / 2
        )

    return mseo_trace, aseo_trace


@njit
def get_mean_std(
    values: NDArray[np.float_], probs: NDArray[np.float_], times: int = 0
) -> tuple[float, float]:
    if times <= 0:
        mean = np.dot(values, probs)
        std = np.sqrt(np.dot((values - mean) ** 2, probs))
    else:
        choices = np.zeros((times,), dtype=np.float_)
        for i in range(times):
            choices[i] = values[
                np.searchsorted(
                    np.cumsum(probs), np.random.random(), side="right"
                )
            ]
        mean = float(np.mean(choices))
        std = float(np.std(choices))
    return mean, std
