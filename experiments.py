import multiprocessing as mp
from multiprocessing.managers import DictProxy
from typing import DefaultDict, Dict, FrozenSet, List, Optional, Tuple, Type

import numpy as np
from matplotlib.figure import Figure
from tqdm.std import tqdm

from ferm import ge, solve_gefair
from tasks import BaseTask

ExpKey = FrozenSet[Tuple[str, float]]
ExpResult = Tuple[DefaultDict[float, int], List[float], float, float]
LThresholdCacheDict = Dict[FrozenSet[float], float]


class MainExperiment:
    def __init__(
        self,
        task_cls: Type[BaseTask],
        thr_finding_granularity: int = 200,
        lambda_quantization_granularity: int = 200,
    ):
        self.task = task_cls()
        self.thr_finding_granularity = thr_finding_granularity
        self.lambda_quantization_granularity = lambda_quantization_granularity

    def _run(
        self,
        param: Dict[str, float],
        L_threshold_cache: LThresholdCacheDict,
    ) -> Tuple[ExpKey, ExpResult]:
        """
        Run experiment with given parameters.
        """

        assert "alpha" in param, "alpha must be in params"
        assert "r" in param, "r must be in params"
        assert "gamma" in param, "gamma must be in params"
        assert "nu" in param, "nu must be in params"
        assert "lambda_max" in param, "lambda_max must be in params"
        assert "exp_index" in param, "exp_index must be in params"
        assert "exp_length" in param, "exp_length must be in params"

        key = frozenset(param.items())

        alpha = param["alpha"]
        r = param["r"]
        gamma = param["gamma"]
        nu = param["nu"]
        lambda_max = param["lambda_max"]
        exp_index = param["exp_index"]
        exp_length = param["exp_length"]

        ######################
        ### Find threshold ###
        ######################

        def get_I_alpha_err_with_threshold(
            threshold: float,
        ) -> Tuple[float, float]:
            """Calculate I_alpha with given threshold"""
            (
                y_hat,
                confusion_matrix,
            ) = self.task.predict_train_with_threshold(threshold)
            [tn, fp], [fn, tp] = confusion_matrix.astype(float)
            err = (fp + fn) / len(y_hat)

            # Calculate I (reference: Appendix A)
            X = ["C", "FN", "FP"]
            b = {"C": r, "FN": r - 1, "FP": r + 1}
            P_x = {
                "C": (tn + tp) / len(y_hat),
                "FN": fn / len(y_hat),
                "FP": fp / len(y_hat),
            }
            I_alpha = ge(alpha, b, X, P_x)

            return I_alpha, err

        def L_threshold(threshold: float, lambda_: float) -> float:
            """Find Lagrangian with given lambda and threshold value"""
            L_threshold_cache_key = frozenset(
                [threshold, lambda_, r, alpha, gamma]
            )
            if L_threshold_cache_key in L_threshold_cache:
                return L_threshold_cache[L_threshold_cache_key]

            I_alpha, err = get_I_alpha_err_with_threshold(threshold)

            # Calculate L (reference: Section 5)
            L = err + lambda_ * (I_alpha - gamma)

            L_threshold_cache[L_threshold_cache_key] = L
            return L

        def find_threshold(lambda_: float) -> float:
            """Find threshold for a lambda value"""
            thresholds = np.linspace(0, 1, self.thr_finding_granularity)

            L_of_lambda = []
            for thr in thresholds:
                L = L_threshold(thr, lambda_)
                L_of_lambda.append(L)
            thr_of_lambda = float(thresholds[np.argmin(L_of_lambda)])

            return thr_of_lambda

        ################
        ### Run FERM ###
        ################

        # Note: lambda quantization is used to reduce the number of times

        lambda_quantization_bins = np.linspace(
            0,
            lambda_max,
            self.lambda_quantization_granularity,
        )

        oracle_cache = []
        for lambda_ in tqdm(
            lambda_quantization_bins,
            desc=f"{exp_index+1} / {exp_length} [1] Generating oracle caches",
        ):
            thr = find_threshold(lambda_)
            oracle_cache.append(thr)

        lagrangian_cache = []
        for lambda_i, lambda_ in tqdm(
            enumerate(lambda_quantization_bins),
            desc=f"{exp_index+1} / {exp_length} [2] Generating lagrangian caches",
        ):
            # lagrangian to be used in ge_fair solver is a function of lambda
            # because it will use $\\hat{h}^{(t)}$ which is a function of lambda
            L = L_threshold(oracle_cache[lambda_i], lambda_)
            lagrangian_cache.append(L)

        def oracle(lambda_: float) -> float:
            """An oracle for FERM solver (finding $\\hat{h}^{(t)}$)"""
            return oracle_cache[
                np.argmin(np.abs(lambda_quantization_bins - lambda_))
            ]

        def lagrangian(h: float, lambda_: float) -> float:
            """Lagrangian for FERM solver (finding $L(h,\\lambda)$)"""
            return lagrangian_cache[
                np.argmin(np.abs(lambda_quantization_bins - lambda_))
            ]

        h_counter, lambda_history = solve_gefair(
            alpha,
            lambda_max,
            nu,
            r,
            gamma,
            lagrangian,
            oracle,
            description=f"{exp_index+1} / {exp_length} [3] Solving",
        )

        #########################
        ### Test I_alpha, err ###
        #########################

        h_prob = {
            h: count / len(lambda_history) for h, count in h_counter.items()
        }

        test_I_alpha = 0.0
        test_err = 0.0
        for h, prob in h_prob.items():
            thr = h
            I_alpha, err = get_I_alpha_err_with_threshold(thr)
            test_I_alpha += I_alpha * prob
            test_err += err * prob

        results_tuple: ExpResult = (
            h_counter,
            lambda_history,
            test_I_alpha,
            test_err,
        )

        return key, results_tuple

    def run(self, params: List[Dict[str, float]]) -> Dict[ExpKey, ExpResult]:
        """Run experiments with given parameters"""

        assert self.task.is_ready(), "Task must be trained and tested first."

        mp_man = mp.Manager()
        L_threshold_cache: DictProxy[FrozenSet[float], float] = mp_man.dict()

        mp_args = []
        for parami, param in enumerate(params):
            mp_param = param.copy()
            mp_param["exp_index"] = parami
            mp_param["exp_length"] = len(params)
            mp_args.append((mp_param, L_threshold_cache))

        results: Dict[ExpKey, ExpResult] = {}
        pool = mp.Pool(processes=max(mp.cpu_count() - 4, 1))
        for key, result in pool.starmap(self._run, mp_args):
            results[key] = result

        return results

    def plot(
        self,
        data: Dict[ExpKey, np.ndarray],
        title: Optional[str] = None,
        legend: Optional[List[str]] = None,
        legend_loc: Optional[str] = None,
        colors: Optional[List[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> List[Figure]:
        """To be implemented"""
        raise NotImplementedError
