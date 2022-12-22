from itertools import product
from typing import Dict, List


def get_params_combination(
    params_dict: Dict[str, List[float]]
) -> List[Dict[str, float]]:
    return [
        dict(zip(params_dict.keys(), v)) for v in product(*params_dict.values())
    ]
