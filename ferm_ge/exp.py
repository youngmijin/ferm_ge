import gc
import warnings
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import numpy as np
import psutil
from numpy.typing import NDArray

from .alg_ge import calc_ge_confmat, calc_ge_v
from .alg_gefair import GEFairSolver
from .alg_rfp_rfn import calc_rfp_rfn
from .exp_param import ParamSet, get_param_sets
from .exp_utils import (
    Cache,
    FakePool,
    get_mean_std,
    get_prob_choices,
    get_time_averaged_trace,
)
from .task_blc import BinaryLogisticClassification

__all__ = ["ExpValidResult", "ExpTrainResult", "run_exp"]


@dataclass
class ExpValidResult:
    ge: tuple[float, float]
    ge_baseline: float
    err: tuple[float, float]
    err_baseline: float

    v: tuple[float, float] | None
    group_size: dict[str, int] | None
    group_confmat: dict[str, tuple[float, float, float, float]] | None
    group_ratio: dict[str, float] | None
    group_rfp: dict[str, tuple[float, float]] | None
    group_rfn: dict[str, tuple[float, float]] | None
    total_rfp: tuple[float, float] | None
    total_rfn: tuple[float, float] | None


@dataclass
class ExpTrainResult:
    ge_bar_trace: NDArray[np.float_] | None
    err_bar_trace: NDArray[np.float_] | None


def run_exp(
    classifier: BinaryLogisticClassification,
    param_dict: dict[str, list[float]],
    keep_trace: bool = True,
    include_valid: bool = True,
    include_group_metrics: bool = False,
    valid_times: int = 0,
    thr_granularity: int = 201,
    no_threading: bool = False,
) -> dict[ParamSet, tuple[ExpTrainResult, ExpValidResult | None]]:
    if not (include_valid or keep_trace):
        warnings.warn(
            "Both include_valid and keep_trace are False. "
            "This will result in no results being returned.",
            UserWarning,
        )

    thr_candidates: list[float] = np.linspace(0, 1, thr_granularity).tolist()

    # pre-calculate error and confmat by threshold
    t_confmat_by_thr_idx: list[tuple[float, float, float, float]] = [
        (0, 0, 0, 0) for _ in thr_candidates
    ]
    t_err_by_thr_idx: list[float] = [0 for _ in thr_candidates]

    for thr_idx, thr in enumerate(thr_candidates):
        _, confmat = classifier.predict_train(thr)
        tn, fp, fn, tp = confmat.astype(float)
        t_confmat_by_thr_idx[thr_idx] = (tn, fp, fn, tp)
        t_err_by_thr_idx[thr_idx] = (fp + fn) / (tn + fp + fn + tp)

    # pre-calculate baseline index for error and ge
    metric_baseline_idx = np.argmin(t_err_by_thr_idx)

    # pre-calculate generalized entropy by alpha/c/a
    t_ge_by_alpaca = Cache[float, list[float]]()

    for alpha in set(param_dict["alpha"]):
        for c in set(param_dict["c"]):
            for a in set(param_dict["a"]):
                ge_list: list[float] = [0 for _ in thr_candidates]

                for thr_idx, thr in enumerate(thr_candidates):
                    tn, fp, fn, tp = t_confmat_by_thr_idx[thr_idx]
                    ge_list[thr_idx] = calc_ge_confmat(
                        alpha, c, a, tn, fp, fn, tp
                    )

                t_ge_by_alpaca.set(ge_list, alpha=alpha, c=c, a=a)

    # build GEFairSolver and save threading arguments with memory consumption
    lib_path = GEFairSolver.compile_gefair(trace_hypi_t=keep_trace)
    solver = GEFairSolver(lib_path)  # to test build and get memory usage

    threading_mem_usage: list[float] = []
    threading_args = []
    for param_set in get_param_sets(param_dict):
        mem_gefair, T = solver.predict_memory_usage(
            len(thr_candidates),
            param_set.lambda_max,
            param_set.nu,
            param_set.alpha,
            param_set.gamma,
            param_set.c,
            param_set.a,
        )

        mem_expresult = 0
        if keep_trace:
            mem_expresult += T * 8 * 2

        threading_mem_usage.append(mem_gefair + mem_expresult)
        threading_args.append((param_set,))

    del solver

    # define runner function
    def runner(
        ps: ParamSet,
    ) -> tuple[ParamSet, ExpTrainResult, ExpValidResult | None]:
        gefair_result = GEFairSolver(lib_path).solve_gefair(
            thr_candidates=thr_candidates,
            I_alpha_cache=t_ge_by_alpaca.get(alpha=ps.alpha, c=ps.c, a=ps.a),
            err_cache=t_err_by_thr_idx,
            lambda_max=ps.lambda_max,
            nu=ps.nu,
            alpha=ps.alpha,
            gamma=ps.gamma,
            c=ps.c,
            a=ps.a,
        )

        # collect training results - (generalized entropy & error trace)
        ge_bar_trace: NDArray[np.float_] | None = None
        err_bar_trace: NDArray[np.float_] | None = None
        if gefair_result.thr_idx_t is not None:
            ge_bar_trace = get_time_averaged_trace(
                gefair_result.thr_idx_t,
                np.array(t_ge_by_alpaca.get(alpha=ps.alpha, c=ps.c, a=ps.a)),
            )
            err_bar_trace = get_time_averaged_trace(
                gefair_result.thr_idx_t, np.array(t_err_by_thr_idx)
            )

        train_result = ExpTrainResult(
            ge_bar_trace=ge_bar_trace,
            err_bar_trace=err_bar_trace,
        )

        # collect validation results if needed
        valid_result: ExpValidResult | None = None
        if include_valid:
            # collect validation results - 1 (generalized entropy & error)
            v_thr_idxs = gefair_result.thr_idx_stat.keys()
            v_thr_probs = (
                np.array(
                    list(gefair_result.thr_idx_stat.values()),
                    dtype=np.float_,
                )
                / gefair_result.T
            )
            v_thr_cnt = len(v_thr_idxs)
            v_thr_choices = get_prob_choices(v_thr_probs, valid_times)

            v_confmat_by_thr_idx = [
                classifier.predict_valid(thr_candidates[thr_idx])[1]
                for thr_idx in v_thr_idxs
            ]
            v_ge_by_thr_idx = np.array(
                [
                    calc_ge_confmat(ps.alpha, ps.c, ps.a, *confmat)
                    for confmat in v_confmat_by_thr_idx
                ]
            )
            v_err_by_thr_idx = np.array(
                [
                    (fp + fn) / (tn + fp + fn + tp)
                    for tn, fp, fn, tp in v_confmat_by_thr_idx
                ]
            )

            v_ge = get_mean_std(v_ge_by_thr_idx, v_thr_probs, v_thr_choices)
            v_err = get_mean_std(v_err_by_thr_idx, v_thr_probs, v_thr_choices)

            v_confmat_baseline = classifier.predict_valid(
                thr_candidates[metric_baseline_idx]
            )[1]
            v_ge_baseline = calc_ge_confmat(
                ps.alpha, ps.c, ps.a, *v_confmat_baseline
            )
            v_err_baseline = (
                v_confmat_baseline[1] + v_confmat_baseline[2]
            ) / sum(v_confmat_baseline)

            # collect testing results - 3 (v & rfp & rfn)
            v_v: tuple[float, float] | None = None
            v_group_confmat: dict[
                str, tuple[float, float, float, float]
            ] | None = None
            v_group_size: dict[str, int] | None = None
            v_group_ratio: dict[str, float] | None = None
            v_group_rfn: dict[str, tuple[float, float]] | None = None
            v_group_rfp: dict[str, tuple[float, float]] | None = None
            v_total_rfn: tuple[float, float] | None = None
            v_total_rfp: tuple[float, float] | None = None

            if include_group_metrics:
                group_cnt = len(classifier.group_names)
                group_size = np.zeros(group_cnt, dtype=np.int_)
                v_v_by_thr_idx = np.zeros(v_thr_cnt, dtype=np.float_)
                v_group_confmat_by_thr_idx = np.zeros(
                    (v_thr_cnt, 4, group_cnt), dtype=np.float_
                )
                v_group_rfp_by_thr_idx = np.zeros(
                    (v_thr_cnt, group_cnt), dtype=np.float_
                )
                v_group_rfn_by_thr_idx = np.zeros(
                    (v_thr_cnt, group_cnt), dtype=np.float_
                )
                v_total_rfp_by_thr_idx = np.zeros(v_thr_cnt, dtype=np.float_)
                v_total_rfn_by_thr_idx = np.zeros(v_thr_cnt, dtype=np.float_)
                for i, thr_idx in enumerate(v_thr_idxs):
                    total_confmat = classifier.predict_valid(
                        thr_candidates[thr_idx]
                    )[1].astype(float)
                    group_confmat = np.zeros((4, group_cnt), dtype=np.float_)
                    for group_idx, group_name in enumerate(
                        classifier.group_names
                    ):
                        _, confmat = classifier.predict_valid(
                            thr_candidates[thr_idx], group=group_name
                        )
                        group_confmat[:, group_idx] = confmat.astype(float)
                        group_size[group_idx] = np.sum(confmat)
                    v_group_confmat_by_thr_idx[i] = group_confmat
                    v_v_by_thr_idx[i] = calc_ge_v(
                        ps.alpha, ps.c, ps.a, *group_confmat
                    )
                    (
                        group_rfp,
                        group_rfn,
                        total_rfp,
                        total_rfn,
                    ) = calc_rfp_rfn(
                        *total_confmat,  # type: ignore
                        *group_confmat,  # type: ignore
                    )
                    v_group_rfp_by_thr_idx[i] = group_rfp
                    v_group_rfn_by_thr_idx[i] = group_rfn
                    v_total_rfp_by_thr_idx[i] = total_rfp
                    v_total_rfn_by_thr_idx[i] = total_rfn
                v_v = get_mean_std(v_v_by_thr_idx, v_thr_probs, v_thr_choices)

                v_group_size = {}
                v_group_confmat = {}
                v_group_ratio = {}
                v_group_rfp = {}
                v_group_rfn = {}
                for group_idx, group_name in enumerate(classifier.group_names):
                    v_group_size[group_name] = int(group_size[group_idx])
                    v_group_confmat[group_name] = tuple(
                        [
                            get_mean_std(
                                v_group_confmat_by_thr_idx[:, i, group_idx],
                                v_thr_probs,
                                v_thr_choices,
                            )[0]
                            for i in range(4)
                        ]
                    )  # type: ignore
                    v_group_ratio[group_name] = group_size[group_idx] / np.sum(
                        group_size
                    )
                    v_group_rfp[group_name] = get_mean_std(
                        v_group_rfp_by_thr_idx[:, group_idx],
                        v_thr_probs,
                        v_thr_choices,
                    )
                    v_group_rfn[group_name] = get_mean_std(
                        v_group_rfn_by_thr_idx[:, group_idx],
                        v_thr_probs,
                        v_thr_choices,
                    )

                v_total_rfp = get_mean_std(
                    v_total_rfp_by_thr_idx, v_thr_probs, v_thr_choices
                )
                v_total_rfn = get_mean_std(
                    v_total_rfn_by_thr_idx, v_thr_probs, v_thr_choices
                )

            valid_result = ExpValidResult(
                ge=v_ge,
                ge_baseline=v_ge_baseline,
                err=v_err,
                err_baseline=v_err_baseline,
                v=v_v,
                group_size=v_group_size,
                group_confmat=v_group_confmat,
                group_ratio=v_group_ratio,
                group_rfp=v_group_rfp,
                group_rfn=v_group_rfn,
                total_rfp=v_total_rfp,
                total_rfn=v_total_rfn,
            )

        del gefair_result

        return ps, train_result, valid_result

    # execute in parallel
    pool_size = min(
        max(1, psutil.cpu_count(logical=False) - 1),
        len(threading_args),
    )
    pool = FakePool() if no_threading else ThreadPool(processes=pool_size)
    results: dict[ParamSet, tuple[ExpTrainResult, ExpValidResult | None]] = {}
    while True:
        # memory-aware parallelization
        mem_available = psutil.virtual_memory().available - 2 * 1024**3
        mem_to_be_used = threading_mem_usage.pop(0)
        args_to_be_used = [threading_args.pop(0)]

        while len(threading_args) > 0:
            if len(args_to_be_used) == pool_size:
                break
            mem_to_be_used_next = threading_mem_usage[0] + mem_to_be_used
            if mem_to_be_used_next > mem_available:
                break
            mem_to_be_used = mem_to_be_used_next
            args_to_be_used.append(threading_args.pop(0))
            threading_mem_usage.pop(0)

        # execute
        for runner_ret in pool.starmap(runner, args_to_be_used):  # type: ignore
            ps: ParamSet = runner_ret[0]
            train_result: ExpTrainResult = runner_ret[1]
            valid_result: ExpValidResult | None = runner_ret[2]

            # save results
            results[ps] = train_result, valid_result

            # finish
            gc.collect()

        if len(threading_args) == 0:
            break

    pool.close()  # type: ignore
    return results
