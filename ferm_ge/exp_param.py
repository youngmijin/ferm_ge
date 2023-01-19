from __future__ import annotations

from dataclasses import dataclass
from itertools import product

__all__ = ["ParamSet", "get_param_sets"]


@dataclass(frozen=True)
class ParamSet:
    lambda_max: float
    nu: float
    alpha: float
    gamma: float
    c: float
    a: float

    @staticmethod
    def from_dict(d: dict[str, float]) -> ParamSet:
        assert "lambda_max" in d, "lambda_max must be in d"
        assert "nu" in d, "nu must be in d"
        assert "alpha" in d, "alpha must be in d"
        assert "gamma" in d, "gamma must be in d"
        assert "c" in d, "c must be in d"
        assert "a" in d, "a must be in d"

        return ParamSet(
            lambda_max=d["lambda_max"],
            nu=d["nu"],
            alpha=d["alpha"],
            gamma=d["gamma"],
            c=d["c"],
            a=d["a"],
        )


def get_param_sets(param_dict: dict[str, list[float]]) -> list[ParamSet]:
    return [
        ParamSet.from_dict(dict(zip(param_dict.keys(), v)))
        for v in product(*param_dict.values())
    ]
