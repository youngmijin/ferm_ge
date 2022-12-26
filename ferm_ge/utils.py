from itertools import product
from typing import Dict, FrozenSet, List, Tuple, Union

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


def predict_memory_consumption(
    thr_finding_granularity: int,
    alpha: float,
    lambda_max: float,
    nu: float,
    r: float,
    gamma: float,
    collect_ge_history: bool,
) -> float:
    """
    Predict memory consumption in bytes during GEFAIR algorithm.
    """

    if alpha == 0:
        I_max = np.log((r + 1) / (r - 1))
    elif alpha == 1:
        I_max = (r + 1) / (r - 1) * np.log((r + 1) / (r - 1))
    else:
        I_max = (np.power((r + 1) / (r - 1), alpha) - 1) / np.abs(
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
