from typing import Generic, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = [
    "Cache",
    "FakePool",
    "get_mean_std",
    "get_prob_choices",
    "get_time_averaged_trace",
]


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
def get_prob_choices(
    probs: NDArray[np.float_], times: int
) -> NDArray[np.intp] | None:
    if times <= 0:
        return None
    choices = np.zeros((times,), dtype=np.intp)
    for i in range(times):
        choices[i] = np.searchsorted(
            np.cumsum(probs), np.random.random(), side="right"
        )
    return choices


@njit
def get_mean_std(
    values: NDArray[np.float_],
    probs: NDArray[np.float_],
    choices: NDArray[np.intp] | None,
) -> tuple[float, float]:
    values = np.ascontiguousarray(values)
    if choices is None:
        mean = np.dot(values, probs)
        std = np.sqrt(np.dot((values - mean) ** 2, probs))
    else:
        chosen_values = values[choices]
        mean = float(np.mean(chosen_values))
        std = float(np.std(chosen_values))
    return mean, std
