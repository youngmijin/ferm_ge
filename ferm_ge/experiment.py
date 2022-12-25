import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import tqdm.std as tqdm

from .algorithm_ge import ge_confmat
from .algorithm_gefair import GEFairResult, GEFairSolver
from .tasks import BaseTask
from .utils import (
    FrozenKey,
    frozenkey_to_paramdict,
    get_params_combination,
    paramdict_to_frozenkey,
)


@dataclass
class Metrics:
    I_alpha: float
    I_alpha_std: Optional[float]
    err: float
    err_std: Optional[float]


class Experiment:
    def __init__(
        self,
        task_cls: Type[BaseTask],
        thr_finding_granularity: int = 200,
    ):
        self.task = task_cls()
        self.thr_candidates: List[float] = np.linspace(
            0, 1, thr_finding_granularity
        ).tolist()

        GEFairSolver.compile_gefair()

    def calc_ge_without_ferm(self, alpha, r) -> Metrics:
        """
        Calculate GE fairness without FERM.
        This is useful to find r.
        """

        assert self.task.trained, "Task must be trained first."

        _, _, (tn, fp, fn, tp) = self.task.predict_train()

        err: float = (fp + fn) / (tn + fp + fn + tp)
        I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)

        return Metrics(I_alpha, None, err, None)

    def _solve(
        self,
        param: Dict[str, float],
        I_alpha_cache: List[float],
        err_cache: List[float],
    ) -> Tuple[FrozenKey, GEFairResult]:
        """
        Solve a FERM-GE problem with given parameters.
        """

        assert "alpha" in param, "alpha must be in params"
        assert "r" in param, "r must be in params"
        assert "gamma" in param, "gamma must be in params"
        assert "nu" in param, "nu must be in params"
        assert "lambda_max" in param, "lambda_max must be in params"

        alpha = param["alpha"]
        r = param["r"]
        gamma = param["gamma"]
        nu = param["nu"]
        lambda_max = param["lambda_max"]

        key = paramdict_to_frozenkey(param)

        result = GEFairSolver().solve_gefair(
            self.thr_candidates,
            I_alpha_cache,
            err_cache,
            alpha,
            lambda_max,
            nu,
            r,
            gamma,
        )

        return key, result

    def solve(
        self, params: Dict[str, List[float]]
    ) -> Dict[FrozenKey, GEFairResult]:
        """Solve the FERM-GE problems with given parameters"""

        assert (
            self.task.trained and self.task.tested
        ), "Task must be trained and tested first."

        params_comb = get_params_combination(params)

        print("Calculating error and GE fairness...", end=" ", flush=True)

        err_confmat_cache: Dict[
            float, Tuple[float, Tuple[float, float, float, float]]
        ] = {}
        for thr in self.thr_candidates:
            _, confmat = self.task.predict_test_with_threshold(thr)
            tn, fp, fn, tp = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            err_confmat_cache[thr] = (err, (tn, fp, fn, tp))

        ge_cache: Dict[float, Dict[float, Tuple[List[float], List[float]]]] = {}
        for alpha in params["alpha"]:
            ge_cache[alpha] = {}
            for r in params["r"]:
                I_alpha_list = []
                err_list = []
                for thr in self.thr_candidates:
                    err, (tn, fp, fn, tp) = err_confmat_cache[thr]
                    I_alpha = ge_confmat(alpha, r, tn, fp, fn, tp)
                    I_alpha_list.append(I_alpha)
                    err_list.append(err)
                ge_cache[alpha][r] = (I_alpha_list, err_list)

        print("done", flush=True)

        mp_args = []
        for param in params_comb:
            mp_args.append((param, *ge_cache[param["alpha"]][param["r"]]))

        results: Dict[FrozenKey, GEFairResult] = {}
        pool = Pool(processes=max(mp.cpu_count() - 4, 1))
        pbar = tqdm.tqdm(total=len(mp_args), desc="Solving FERM-GE problems...")
        for key, result in pool.starmap(self._solve, mp_args):
            results[key] = result
            pbar.update()
        pbar.close()

        return results

    def get_metrics_with_prob(
        self,
        exp_results: Dict[FrozenKey, GEFairResult],
    ) -> Dict[FrozenKey, Metrics]:
        """Get metrics with probability"""

        exp_metrics: Dict[FrozenKey, Metrics] = {}

        for exp_key, exp_result in exp_results.items():
            param_dict = frozenkey_to_paramdict(exp_key)
            alpha = param_dict["alpha"]
            r = param_dict["r"]

            thr_prob = {
                thr: count / exp_result.T
                for thr, count in exp_result.D_bar.items()
            }

            test_I_alpha = 0.0
            test_err = 0.0
            for thr, prob in thr_prob.items():
                confmat = self.task.predict_test_with_threshold(thr)[1]
                tn, fp, fn, tp = confmat.astype(float)
                err: float = (fp + fn) / (tn + fp + fn + tp)
                I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)
                test_I_alpha += I_alpha * prob
                test_err += err * prob

            exp_metrics[exp_key] = Metrics(
                I_alpha=test_I_alpha,
                I_alpha_std=None,
                err=test_err,
                err_std=None,
            )

        return exp_metrics

    def _get_metrics_with_repeat(
        self,
        exp_key: FrozenKey,
        exp_result: GEFairResult,
        repeat: int,
    ) -> Tuple[FrozenKey, Metrics]:
        param_dict = frozenkey_to_paramdict(exp_key)
        alpha = param_dict["alpha"]
        r = param_dict["r"]

        thr_prob = {
            thr: count / exp_result.T for thr, count in exp_result.D_bar.items()
        }
        rand_items = list(thr_prob.keys())
        rand_probs = list(thr_prob.values())

        ge_err_cache: Dict[float, Tuple[float, float]] = {}
        for thr in rand_items:
            confmat = self.task.predict_test_with_threshold(thr)[1]
            tn, fp, fn, tp = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)
            ge_err_cache[thr] = (I_alpha, err)

        test_I_alpha = []
        test_err = []
        for _ in range(repeat):
            rand_thr = np.random.choice(rand_items, p=rand_probs)
            I_alpha, err = ge_err_cache[rand_thr]
            test_I_alpha.append(I_alpha)
            test_err.append(err)

        np_I_alpha = np.array(test_I_alpha)
        np_err = np.array(test_err)

        return exp_key, Metrics(
            I_alpha=np_I_alpha.mean(),
            I_alpha_std=np_I_alpha.std(),
            err=np_err.mean(),
            err_std=np_err.std(),
        )

    def get_metrics_with_repeat(
        self,
        exp_results: Dict[FrozenKey, GEFairResult],
        repeat: int = 10000,
    ) -> Dict[FrozenKey, Metrics]:
        """Get metrics with repating"""

        mp_args = []
        for key, result in exp_results.items():
            mp_args.append((key, result, repeat))

        exp_metrics: Dict[FrozenKey, Metrics] = {}
        pool = mp.Pool(processes=max(mp.cpu_count() - 4, 1))
        for key, metric in pool.starmap(self._get_metrics_with_repeat, mp_args):
            exp_metrics[key] = metric

        return exp_metrics
