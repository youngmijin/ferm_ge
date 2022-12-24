import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    Type,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .algorithms import ge_confmat, solve_gefair
from .tasks import BaseTask
from .utils import get_params_combination

ExperimentKey = FrozenSet[Tuple[str, float]]
GEErrCacheDict = Dict[FrozenSet[float], Tuple[float, float]]


@dataclass
class ExperimentResult:
    h_counter: DefaultDict[float, int]
    lambda_history: List[float]


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
        self.thr_finding_granularity = thr_finding_granularity

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
        ge_err_cache: GEErrCacheDict,
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

        def calc_L(threshold: float, lambda_: float) -> float:
            """Find Lagrangian with given lambda and threshold"""

            I_alpha, err = ge_err_cache[frozenset([alpha, r, threshold])]
            return err + lambda_ * (I_alpha - gamma)

        thr_candidates = np.linspace(0, 1, self.thr_finding_granularity)
        thr_cache: Dict[float, float] = {}

        def find_threshold(lambda_: float) -> float:
            """Find threshold for a given lambda value"""

            if lambda_ in thr_cache:
                return thr_cache[lambda_]

            thr_of_lambda = float(
                thr_candidates[
                    np.argmin([calc_L(thr, lambda_) for thr in thr_candidates])
                ]
            )

            thr_cache[lambda_] = thr_of_lambda
            return thr_of_lambda

        #############################
        ### Solve FERM-GE Problem ###
        #############################

        h_counter, lambda_history = solve_gefair(
            alpha,
            lambda_max,
            nu,
            r,
            gamma,
            calc_L,
            find_threshold,
            show_progress=f"{exp_name} Solving a fairness problem",
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

        assert (
            self.task.trained and self.task.tested
        ), "Task must be trained and tested first."

        params_comb = get_params_combination(params)

        thr_candidates = np.linspace(0, 1, self.thr_finding_granularity)
        err_confmat_cache: Dict[
            float, Tuple[float, Tuple[float, float, float, float]]
        ] = {}
        for thr in thr_candidates:
            _, confmat = self.task.predict_test_with_threshold(thr)
            tn, fp, fn, tp = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            err_confmat_cache[thr] = (err, (tn, fp, fn, tp))

        ge_err_cache: GEErrCacheDict = {}
        for alpha in params["alpha"]:
            for r in params["r"]:
                for thr in thr_candidates:
                    ge_err_cache_key = frozenset([alpha, r, thr])
                    err, (tn, fp, fn, tp) = err_confmat_cache[thr]
                    I_alpha = ge_confmat(alpha, r, tn, fp, fn, tp)
                    ge_err_cache[ge_err_cache_key] = (I_alpha, err)

        mp_args = []
        for parami, param in enumerate(params_comb):
            mp_param = param.copy()
            mp_param["exp_index"] = parami
            mp_param["exp_total"] = len(params_comb)
            mp_args.append((mp_param, ge_err_cache))

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
        exp_key: ExperimentKey,
        exp_result: ExperimentResult,
        repeat: int,
    ) -> Tuple[ExperimentKey, Metrics]:
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

        test_I_alpha = []
        test_err = []
        for _ in range(repeat):
            rand_h = np.random.choice(rand_items, p=rand_probs)
            thr = rand_h
            confmat = self.task.predict_test_with_threshold(thr)[1]
            tn, fp, fn, tp = confmat.astype(float)
            err: float = (fp + fn) / (tn + fp + fn + tp)
            I_alpha: float = ge_confmat(alpha, r, tn, fp, fn, tp)
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
        exp_results: Dict[ExperimentKey, ExperimentResult],
        repeat: int = 10000,
    ) -> Dict[ExperimentKey, Metrics]:
        """Get metrics with repating"""

        mp_args = []
        for key, result in exp_results.items():
            mp_args.append((key, result, repeat))

        exp_metrics: Dict[ExperimentKey, Metrics] = {}
        pool = mp.Pool(processes=max(mp.cpu_count() - 4, 1))
        for key, metric in pool.starmap(self._get_metrics_with_repeat, mp_args):
            exp_metrics[key] = metric

        return exp_metrics

    def plot_metrics(
        self,
        exp_metrics: Dict[ExperimentKey, Metrics],
        metric_name: str,
        params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
        figsize: Tuple[float, float] = (8, 6),
        save_path: Optional[str] = None,
    ) -> Figure:
        """Plot metrics"""

        assert metric_name in ["I_alpha", "err"]

        experiments_to_draw: Dict[ExperimentKey, Metrics] = {}
        r_values = set()
        for exp_key, exp_metric in exp_metrics.items():
            param_dict = {k: v for k, v in list(exp_key)}
            if not params_filter(param_dict):
                continue
            experiments_to_draw[exp_key] = exp_metric
            r_values.add(param_dict["r"])

        fig, ax = plt.subplots(figsize=figsize)

        if len(experiments_to_draw) == 0:
            print("No metrics to draw.")
            return fig

        for r in r_values:
            plot_x = []
            plot_y = []
            plot_err = []
            for exp_key, exp_metric in experiments_to_draw.items():
                param_dict = {k: v for k, v in list(exp_key)}
                if param_dict["r"] != r:
                    continue
                plot_x.append(param_dict["gamma"])
                plot_y.append(getattr(exp_metric, metric_name))
                if getattr(exp_metric, f"{metric_name}_std") is not None:
                    plot_err.append(getattr(exp_metric, f"{metric_name}_std"))
            if len(plot_err) > 0:
                ax.errorbar(plot_x, plot_y, yerr=plot_err, label=f"r={r}")
            else:
                ax.plot(plot_x, plot_y, "o-", label=f"r={r}")

        if len(r_values) > 1:
            ax.legend()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=600)

        return fig
