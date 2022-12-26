import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from typing import Dict, Optional, Tuple

import numpy as np

from .algorithm_ge import ge_confmat
from .algorithm_gefair import GEFairResultSM
from .tasks import BaseTask
from .utils import FrozenKey, frozenkey_to_paramdict


@dataclass
class Metrics:
    I_alpha: float
    I_alpha_std: Optional[float]
    err: float
    err_std: Optional[float]


def calc_metrics(
    exp_results: Dict[FrozenKey, GEFairResultSM],
    task: BaseTask,
    repeat: int = 10000,
) -> Dict[FrozenKey, Metrics]:
    if repeat <= 0:
        return _calc_metrics_with_prob(exp_results, task)
    else:
        mp_args = []
        for key, result in exp_results.items():
            mp_args.append((key, result, task, repeat))

        exp_metrics: Dict[FrozenKey, Metrics] = {}
        pool = Pool(processes=min(max(mp.cpu_count() - 4, 1), len(mp_args)))
        for key, metric in pool.starmap(_calc_metrics_with_repeat, mp_args):
            exp_metrics[key] = metric

        return exp_metrics


def _calc_metrics_with_prob(
    exp_results: Dict[FrozenKey, GEFairResultSM],
    task: BaseTask,
) -> Dict[FrozenKey, Metrics]:
    """Get metrics with probability"""

    exp_metrics: Dict[FrozenKey, Metrics] = {}

    for exp_key, exp_result in exp_results.items():
        param_dict = frozenkey_to_paramdict(exp_key)
        alpha = param_dict["alpha"]
        r = param_dict["r"]

        thr_prob = {
            thr: count / exp_result.T for thr, count in exp_result.D_bar.items()
        }

        test_I_alpha = 0.0
        test_err = 0.0
        for thr, prob in thr_prob.items():
            confmat = task.predict_test_with_threshold(thr)[1]
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


def _calc_metrics_with_repeat(
    exp_key: FrozenKey,
    exp_result: GEFairResultSM,
    task: BaseTask,
    repeat: int,
) -> Tuple[FrozenKey, Metrics]:
    """Get metrics with repeating and sampling"""

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
        confmat = task.predict_test_with_threshold(thr)[1]
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
