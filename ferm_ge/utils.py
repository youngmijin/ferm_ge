from itertools import product
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np

FrozenKey = FrozenSet[Tuple[str, float]]


def get_params_combination(
    params_dict: Dict[str, List[float]]
) -> List[Dict[str, float]]:
    return [
        dict(zip(params_dict.keys(), v)) for v in product(*params_dict.values())
    ]


def frozenkey_to_paramdict(frozenkey: FrozenKey) -> Dict[str, float]:
    return {k: v for k, v in list(frozenkey)}


def paramdict_to_frozenkey(paramdict: Dict[str, float]) -> FrozenKey:
    return frozenset(paramdict.items())


def average_by_time(array: Union[np.ndarray, List[float]]) -> np.ndarray:
    return np.divide(np.cumsum(array), np.arange(len(array)) + 1.0)


def apply_sampling(
    array: np.ndarray,
    sampling_threshold: Optional[int],
    sampling_exclude_initial: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    assert array.ndim == 1, "array must be 1D array"
    if sampling_threshold is None or len(array) <= sampling_threshold:
        return np.arange(len(array)), array
    else:
        former_x = np.arange(sampling_exclude_initial)
        former_y = array[:sampling_exclude_initial]
        latter_x = np.arange(
            sampling_exclude_initial,
            len(array),
            len(array) // sampling_threshold,
        )
        latter_y = array[
            sampling_exclude_initial :: len(array) // sampling_threshold
        ]
        return (
            np.concatenate([former_x, latter_x]),
            np.concatenate([former_y, latter_y]),
        )


def predict_memory_consumption(
    thr_finding_granularity: int,
    alpha: float,
    lambda_max: float,
    nu: float,
    c: float,
    a: float,
    gamma: float,
    collect_ge_history: bool,
) -> float:
    """
    Predict memory consumption in bytes during GEFAIR algorithm.
    """

    if alpha == 0:
        I_max = np.log((c + a) / (c - a))
    elif alpha == 1:
        I_max = (c + a) / (c - a) * np.log((c + a) / (c - a))
    else:
        I_max = (np.power((c + a) / (c - a), alpha) - 1) / np.abs(
            (alpha - 1) * alpha
        )
    A_alpha = 1 + lambda_max * (gamma + I_max)
    T = 4 * A_alpha * A_alpha * np.log(2) / (nu * nu)

    mem_consumption = 0.0

    mem_consumption += 8 * thr_finding_granularity  # mem_thr_cache
    mem_consumption += 8 * thr_finding_granularity  # mem_w0_mult_cache
    mem_consumption += 8 * thr_finding_granularity  # mem_w1_mult_cache

    mem_consumption += 8 * T  # mem_hyp_hist
    mem_consumption += 8 * T  # mem_lbar_hist
    mem_consumption += 8 * thr_finding_granularity  # mem_Dbar_hist

    if collect_ge_history:
        mem_consumption += 8 * T  # mem_ge_hist
        mem_consumption += 8 * T  # mem_err_hist

    return mem_consumption
