import gc
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import numpy as np
import psutil
from numpy.typing import NDArray

from .alg_ge import calc_ge_confmat, calc_ge_v
from .alg_gefair import GEFairResult, GEFairSolver
from .alg_seo import calc_seo
from .exp_param import ParamSet, get_param_sets
from .exp_utils import Cache, FakePool, get_mean_std, get_time_averaged_trace
from .task_blc import BinaryLogisticClassification


@dataclass
class ExpValidResult:
    ge: float
    ge_std: float
    err: float
    err_std: float

    v: float | None
    v_std: float | None
    mseo: float | None
    mseo_std: float | None
    aseo: float | None
    aseo_std: float | None


@dataclass
class ExpTrainResult:
    ge_baseline: float
    err_baseline: float

    ge_bar_trace: NDArray[np.float_] | None
    err_bar_trace: NDArray[np.float_] | None

    mseo_trace: NDArray[np.float_] | None
    aseo_trace: NDArray[np.float_] | None


def run_exp(
    classifier: BinaryLogisticClassification,
    param_dict: dict[str, list[float]],
    keep_trace: bool = True,
    calc_train_seo: bool = False,
    calc_valid_seo: bool = False,
    seo_components: list[str] = ["fp", "fn"],
    include_valid: bool = True,
    valid_times: int = 0,
    thr_granularity: int = 200,
    no_threading: bool = False,
) -> dict[ParamSet, tuple[ExpTrainResult, ExpValidResult | None]]:
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

    # pre-calculate seo by threshold if needed
    t_mseo_by_thr_idx: NDArray[np.float_] | None = None
    t_aseo_by_thr_idx: NDArray[np.float_] | None = None

    group_cnt = len(classifier.group_names)
    if calc_train_seo:
        t_mseo_by_thr_idx = np.zeros(thr_granularity, dtype=np.float_)
        t_aseo_by_thr_idx = np.zeros(thr_granularity, dtype=np.float_)
        for thr_idx, thr in enumerate(thr_candidates):
            group_confmat = np.zeros((4, group_cnt), dtype=np.float_)
            for group_idx, group_name in enumerate(classifier.group_names):
                _, confmat = classifier.predict_train(thr, group=group_name)
                group_confmat[:, group_idx] = confmat.astype(float)
            mseo, aseo = calc_seo(  # type: ignore
                *t_confmat_by_thr_idx[thr_idx],
                *group_confmat,
                exclude_rfp=(not "fp" in seo_components),
                exclude_rfn=(not "fn" in seo_components),
            )
            t_mseo_by_thr_idx[thr_idx] = mseo
            t_aseo_by_thr_idx[thr_idx] = aseo

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
            if calc_train_seo:
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

        # collect training results - 1 (generalized entropy & error trace)
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

        # collect training results - 2 (mseo & aseo trace)
        mseo_trace: NDArray[np.float_] | None = None
        aseo_trace: NDArray[np.float_] | None = None
        if calc_train_seo and gefair_result.thr_idx_t is not None:
            assert t_mseo_by_thr_idx is not None
            assert t_aseo_by_thr_idx is not None
            mseo_trace = t_mseo_by_thr_idx[gefair_result.thr_idx_t]
            aseo_trace = t_aseo_by_thr_idx[gefair_result.thr_idx_t]

        train_result = ExpTrainResult(
            ge_baseline=t_ge_by_alpaca.get(alpha=ps.alpha, c=ps.c, a=ps.a)[
                metric_baseline_idx
            ],
            err_baseline=t_err_by_thr_idx[metric_baseline_idx],
            ge_bar_trace=ge_bar_trace,
            err_bar_trace=err_bar_trace,
            mseo_trace=mseo_trace,
            aseo_trace=aseo_trace,
        )

        # collect validation results if needed
        valid_result: ExpValidResult | None = None
        if include_valid:
            # collect validation results - 1 (generalized entropy & error)
            result_thr_idxs = gefair_result.thr_idx_stat.keys()
            result_thr_probs = (
                np.array(
                    list(gefair_result.thr_idx_stat.values()),
                    dtype=np.float_,
                )
                / gefair_result.T
            )

            v_confmat_by_thr_idx = [
                classifier.predict_valid(thr_candidates[thr_idx])[1]
                for thr_idx in result_thr_idxs
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

            v_ge_mean, v_ge_std = get_mean_std(
                v_ge_by_thr_idx, result_thr_probs, times=valid_times
            )
            v_err_mean, v_err_std = get_mean_std(
                v_err_by_thr_idx, result_thr_probs, times=valid_times
            )

            # collect testing results - 3 (v & mseo & aseo)
            v_v_mean: float | None = None
            v_v_std: float | None = None
            v_mseo_mean: float | None = None
            v_mseo_std: float | None = None
            v_aseo_mean: float | None = None
            v_aseo_std: float | None = None

            if calc_valid_seo:
                v_v_by_thr_idx = np.zeros(len(result_thr_idxs), dtype=np.float_)
                v_mseo_by_thr_idx = np.zeros(
                    len(result_thr_idxs), dtype=np.float_
                )
                v_aseo_by_thr_idx = np.zeros(
                    len(result_thr_idxs), dtype=np.float_
                )
                for i, thr_idx in enumerate(result_thr_idxs):
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
                    v_v_by_thr_idx[i] = calc_ge_v(
                        ps.alpha, ps.c, ps.a, *group_confmat
                    )
                    mseo, aseo = calc_seo(  # type: ignore
                        *total_confmat,
                        *group_confmat,
                        exclude_rfp=(not "fp" in seo_components),
                        exclude_rfn=(not "fn" in seo_components),
                    )
                    v_mseo_by_thr_idx[i] = mseo
                    v_aseo_by_thr_idx[i] = aseo
                v_v_mean, v_v_std = get_mean_std(
                    v_v_by_thr_idx, result_thr_probs, times=valid_times
                )
                v_mseo_mean, v_mseo_std = get_mean_std(
                    v_mseo_by_thr_idx, result_thr_probs, times=valid_times
                )
                v_aseo_mean, v_aseo_std = get_mean_std(
                    v_aseo_by_thr_idx, result_thr_probs, times=valid_times
                )

            valid_result = ExpValidResult(
                ge=v_ge_mean,
                ge_std=v_ge_std,
                err=v_err_mean,
                err_std=v_err_std,
                v=v_v_mean,
                v_std=v_v_std,
                mseo=v_mseo_mean,
                mseo_std=v_mseo_std,
                aseo=v_aseo_mean,
                aseo_std=v_aseo_std,
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
