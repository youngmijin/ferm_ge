from itertools import product
from typing import Dict, FrozenSet, List, Tuple

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
