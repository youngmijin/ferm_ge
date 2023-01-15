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


class FakePool:
    def __init__(self, *args, **kwargs):
        pass

    def starmap(self, fn, args_list):
        for args in args_list:
            yield fn(*args)

    def close(self):
        pass


@njit
def get_time_averaged_trace(
    hypi_t: NDArray[np.intp],
    cache_by_hypi: NDArray[np.float_],
) -> NDArray[np.float_]:
    """Calculates time-averaged trace (history of bar-values)."""

    assert hypi_t.ndim == 1
    assert cache_by_hypi.ndim == 1

    trace_cumsum: NDArray[np.float_] = np.cumsum(cache_by_hypi[hypi_t])
    time = np.arange(1, len(hypi_t) + 1, dtype=np.float_)

    return trace_cumsum / time


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
