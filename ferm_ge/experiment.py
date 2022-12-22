import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.managers import DictProxy
from typing import DefaultDict, Dict, FrozenSet, List, Tuple, Type

import numpy as np
from tqdm.std import tqdm

from .algorithms import ge_confmat, solve_gefair
from .tasks import BaseTask
from .utils import get_params_combination

ExperimentKey = FrozenSet[Tuple[str, float]]
LThresholdCacheDict = Dict[FrozenSet[float], float]


@dataclass
class ExperimentResult:
    h_counter: DefaultDict[float, int]
    lambda_history: List[float]


@dataclass
class Metrics:
    I_alpha: float
    err: float


class Experiment:
    def __init__(
        self,
        task_cls: Type[BaseTask],
        thr_finding_granularity: int = 200,
        lambda_quantization_granularity: int = 200,
    ):
        self.task = task_cls()
        self.thr_finding_granularity = thr_finding_granularity
        self.lambda_quantization_granularity = lambda_quantization_granularity

    def _solve(
        self,
        param: Dict[str, float],
        L_threshold_cache: LThresholdCacheDict,
    ) -> Tuple[ExperimentKey, ExperimentResult]:
        """
        Solve a FERM-GE problem with given parameters.
        """

        assert "alpha" in param, "alpha must be in params"
        assert "r" in param, "r must be in params"
        assert "gamma" in param, "gamma must be in params"
        assert "nu" in param, "nu must be in params"
        assert "lambda_max" in param, "lambda_max must be in params"
        assert "exp_index" in param, "exp_index must be in params"
        assert "exp_total" in param, "exp_total must be in params"

        key = frozenset(param.items())

        alpha = param["alpha"]
        r = param["r"]
        gamma = param["gamma"]
        nu = param["nu"]
        lambda_max = param["lambda_max"]

        exp_index = int(param["exp_index"])
        exp_total = int(param["exp_total"])
        exp_name = f"{exp_index + 1:04d}/{exp_total:04d}"

        ###################################
        ### Threshold-finding functions ###
        ###################################

        def find_L(threshold: float, lambda_: float) -> float:
            """Find Lagrangian with given lambda and threshold"""

            L_threshold_cache_key = frozenset(
                [threshold, lambda_, r, alpha, gamma]
            )
            if L_threshold_cache_key in L_threshold_cache:
                return L_threshold_cache[L_threshold_cache_key]

            confmat = self.task.predict_train_with_threshold(threshold)[1]
            [tn, fp], [fn, tp] = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)

            # Calculate L (reference: Section 5)
            L = err + lambda_ * (I_alpha - gamma)

            L_threshold_cache[L_threshold_cache_key] = L
            return L

        def find_threshold(lambda_: float) -> float:
            """Find threshold for a given lambda value"""

            thr_candidates = np.linspace(0, 1, self.thr_finding_granularity)

            L_of_lambda = []
            for thr in thr_candidates:
                L = find_L(thr, lambda_)
                L_of_lambda.append(L)

            thr_of_lambda = float(thr_candidates[np.argmin(L_of_lambda)])
            return thr_of_lambda

        ###########################################
        ### Pre-calculate Oracles & Lagrangians ###
        ###########################################

        # Note: lambda quantization is used to reduce the number of times
        #       we will call the quantized lambda as "lambda_q".

        lambda_quantization_bins = np.linspace(
            0,
            lambda_max,
            self.lambda_quantization_granularity + 1,
        )

        oracle_cache: List[float] = []
        for lambda_ in tqdm(
            lambda_quantization_bins,
            desc=f"{exp_name} [1] Generating oracle caches",
        ):
            thr = find_threshold(lambda_)
            oracle_cache.append(thr)

        lagrangian_cache: List[float] = []
        for lambda_i, lambda_ in tqdm(
            enumerate(lambda_quantization_bins),
            desc=f"{exp_name} [2] Generating lagrangian caches",
        ):
            L = find_L(oracle_cache[lambda_i], lambda_)
            lagrangian_cache.append(L)

        #############################
        ### Solve FERM-GE Problem ###
        #############################

        def get_lambda_q_idx(lambda_: float) -> np.intp:
            """Return the nearest lambda's index in lambda quantization bins"""

            return np.argmin(np.abs(lambda_quantization_bins - lambda_))

        def oracle_q(lambda_: float) -> float:
            """An oracle with quantized lambdas (finding $\\hat{h}^{(t)}$)"""

            return oracle_cache[get_lambda_q_idx(lambda_)]

        def get_L_q(h: float, lambda_: float) -> float:
            """Lagrangian with quantized lambdas (finding $L(h,\\lambda)$)"""

            return lagrangian_cache[get_lambda_q_idx(lambda_)]

        h_counter, lambda_history = solve_gefair(
            alpha,
            lambda_max,
            nu,
            r,
            gamma,
            get_L_q,
            oracle_q,
            show_progress=f"{exp_name} [3] Solving",
        )

        result = ExperimentResult(
            h_counter=h_counter,
            lambda_history=lambda_history,
        )

        return key, result

    def solve(
        self, params: Dict[str, List[float]]
    ) -> Dict[ExperimentKey, ExperimentResult]:
        """Solve the FERM-GE problems with given parameters"""

        assert self.task.is_ready(), "Task must be trained and tested first."

        params_comb = get_params_combination(params)

        mp_man = mp.Manager()
        L_threshold_cache: DictProxy[FrozenSet[float], float] = mp_man.dict()

        mp_args = []
        for parami, param in enumerate(params_comb):
            mp_param = param.copy()
            mp_param["exp_index"] = parami
            mp_param["exp_total"] = len(params_comb)
            mp_args.append((mp_param, L_threshold_cache))

        results: Dict[ExperimentKey, ExperimentResult] = {}
        pool = mp.Pool(processes=max(mp.cpu_count() - 4, 1))
        for key, result in pool.starmap(self._solve, mp_args):
            results[key] = result

        return results

    def get_metrics_with_prob(
        self,
        exp_results: Dict[ExperimentKey, ExperimentResult],
    ) -> Dict[ExperimentKey, Metrics]:
        """Get metrics with probability"""

        exp_metrics: Dict[ExperimentKey, Metrics] = {}

        for exp_key, exp_result in exp_results.items():
            param_dict = {k: v for k, v in list(exp_key)}
            alpha = param_dict["alpha"]
            r = param_dict["r"]

            h_counter = exp_result.h_counter
            lambda_history = exp_result.lambda_history

            h_prob = {
                h: count / len(lambda_history) for h, count in h_counter.items()
            }

            test_I_alpha = 0.0
            test_err = 0.0
            for h, prob in h_prob.items():
                thr = h
                confmat = self.task.predict_test_with_threshold(thr)[1]
                [tn, fp], [fn, tp] = confmat.astype(float)
                err: float = (fp + fn) / (tn + fp + fn + tp)
                I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)
                test_I_alpha += I_alpha * prob
                test_err += err * prob

            exp_metrics[exp_key] = Metrics(
                I_alpha=test_I_alpha,
                err=test_err,
            )

        return exp_metrics

    def get_metrics_with_repeat(
        self,
        exp_results: Dict[ExperimentKey, ExperimentResult],
        repeat: int,
    ) -> Dict[ExperimentKey, Metrics]:
        """Get metrics with repating"""

        exp_metrics: Dict[ExperimentKey, Metrics] = {}

        for exp_key, exp_result in tqdm(exp_results.items()):
            param_dict = {k: v for k, v in list(exp_key)}
            alpha = param_dict["alpha"]
            r = param_dict["r"]

            h_counter = exp_result.h_counter
            lambda_history = exp_result.lambda_history

            h_prob = {
                h: count / len(lambda_history) for h, count in h_counter.items()
            }
            rand_items = list(h_prob.keys())
            rand_probs = list(h_prob.values())

            test_I_alpha = 0.0
            test_err = 0.0
            for _ in range(repeat):
                rand_h = np.random.choice(rand_items, p=rand_probs)
                thr = rand_h
                confmat = self.task.predict_test_with_threshold(thr)[1]
                [tn, fp], [fn, tp] = confmat.astype(float)
                err: float = (fp + fn) / (tn + fp + fn + tp)
                I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)
                test_I_alpha += I_alpha
                test_err += err

            test_I_alpha /= repeat
            test_err /= repeat

            exp_metrics[exp_key] = Metrics(
                I_alpha=test_I_alpha,
                err=test_err,
            )

        return exp_metrics
