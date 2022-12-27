import gc
import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool as Pool
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import psutil

from .algorithm_ge import ge_confmat
from .algorithm_gefair import GEFairResultSM, GEFairSolverC
from .metrics import Metrics, calc_metrics
from .tasks import BaseTask
from .utils import (
    FrozenKey,
    get_params_combination,
    paramdict_to_frozenkey,
    predict_memory_consumption,
)


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

        self.lib_path = GEFairSolverC.compile_gefair()

    def __del__(self):
        if os.path.exists(self.lib_path):
            os.remove(self.lib_path)

    def _run(
        self,
        param: Dict[str, float],
        collect_ge_history: bool,
        I_alpha_cache: List[float],
        err_cache: List[float],
    ) -> Tuple[FrozenKey, GEFairResultSM]:
        """
        Run a FERM-GE solver with given parameters.
        """

        assert "alpha" in param, "alpha must be in params"
        assert "c" in param, "c must be in params"
        assert "a" in param, "a must be in params"
        assert "gamma" in param, "gamma must be in params"
        assert "nu" in param, "nu must be in params"
        assert "lambda_max" in param, "lambda_max must be in params"

        alpha = param["alpha"]
        c = param["c"]
        a = param["a"]
        gamma = param["gamma"]
        nu = param["nu"]
        lambda_max = param["lambda_max"]

        key = paramdict_to_frozenkey(param)

        result = GEFairSolverC(self.lib_path).solve_gefair(
            self.thr_candidates,
            I_alpha_cache,
            err_cache,
            alpha,
            lambda_max,
            nu,
            c,
            a,
            gamma,
            collect_ge_history,
        )

        return key, result

    def run(
        self,
        params: Dict[str, List[float]],
        collect_ge_history: bool = False,
        delete_results: bool = False,
        return_metrics: bool = False,
        metrics_repeat_times: int = 10000,
    ) -> Tuple[
        Optional[Dict[FrozenKey, GEFairResultSM]],
        Optional[Dict[FrozenKey, Metrics]],
    ]:
        """Run the FERM-GE solver with given parameters"""

        assert (
            self.task.trained and self.task.tested
        ), "Task must be trained and tested first."

        params_comb = get_params_combination(params)

        err_confmat_cache: Dict[
            float, Tuple[float, Tuple[float, float, float, float]]
        ] = {}
        for thr in self.thr_candidates:
            _, confmat = self.task.predict_test_with_threshold(thr)
            tn, fp, fn, tp = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            err_confmat_cache[thr] = (err, (tn, fp, fn, tp))

        ge_cache: Dict[
            float, Dict[float, Dict[float, Tuple[List[float], List[float]]]]
        ] = {}
        for alpha in set(params["alpha"]):
            ge_cache[alpha] = {}
            for c in set(params["c"]):
                ge_cache[alpha][c] = {}
                for a in set(params["a"]):
                    I_alpha_list = []
                    err_list = []
                    for thr in self.thr_candidates:
                        err, (tn, fp, fn, tp) = err_confmat_cache[thr]
                        I_alpha = ge_confmat(alpha, c, a, tn, fp, fn, tp)
                        I_alpha_list.append(I_alpha)
                        err_list.append(err)
                    ge_cache[alpha][c][a] = (I_alpha_list, err_list)

        mem_usages = []
        mp_args = []
        for param in params_comb:
            mp_args.append(
                (
                    param,
                    collect_ge_history,
                    *ge_cache[param["alpha"]][param["c"]][param["a"]],
                )
            )
            mem_usages.append(
                predict_memory_consumption(
                    len(self.thr_candidates),
                    param["alpha"],
                    param["lambda_max"],
                    param["nu"],
                    param["c"],
                    param["a"],
                    param["gamma"],
                    collect_ge_history,
                )
            )

        if len(mp_args) == 1:
            key, result = self._run(*mp_args[0])
            metric: Optional[Dict[FrozenKey, Metrics]] = None
            if return_metrics:
                metric = calc_metrics(
                    {key: result},
                    self.task,
                    metrics_repeat_times,
                )
            if delete_results:
                del result
                return None, metric
            return {key: result}, metric
        else:
            results: Dict[FrozenKey, GEFairResultSM] = {}
            metrics: Dict[FrozenKey, Metrics] = {}

            pool_max_size = max(mp.cpu_count() - 4, 1)

            while True:
                available_mem = (
                    psutil.virtual_memory().available - 2 * 1024**3
                )
                current_predict_mem = mem_usages.pop(0)
                current_mp_args = [mp_args.pop(0)]
                while len(mp_args) > 0:
                    if len(current_mp_args) == pool_max_size:
                        break
                    next_predict_mem = current_predict_mem + mem_usages[0]
                    if next_predict_mem > available_mem:
                        break
                    current_predict_mem = next_predict_mem
                    current_mp_args.append(mp_args.pop(0))
                    mem_usages.pop(0)

                pool = Pool(processes=len(current_mp_args))
                for key, result in pool.starmap(self._run, current_mp_args):
                    if return_metrics:
                        metrics[key] = list(
                            calc_metrics(
                                {key: result}, self.task, metrics_repeat_times
                            ).values()
                        )[0]
                    if delete_results:
                        del result
                    else:
                        results[key] = result
                pool.close()
                gc.collect()

                if len(mp_args) == 0:
                    break

            if return_metrics and delete_results:
                return None, metrics
            elif not return_metrics and delete_results:
                return None, None
            elif return_metrics and not delete_results:
                return results, metrics
            else:
                return results, None
