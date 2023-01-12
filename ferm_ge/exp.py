import gc
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool

import numpy as np
import psutil
from numpy.typing import NDArray

from .alg_ge import ge_confmat
from .alg_gefair import GEFairResult, GEFairSolver
from .exp_param import ParamSet, get_param_sets
from .exp_utils import (
    Cache,
    get_mean_std,
    get_seo_trace,
    get_time_averaged_trace,
)
from .task_blc import BinaryLogisticClassification


@dataclass
class ExpTestResult:
    ge: float
    ge_std: float
    err: float
    err_std: float

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
    with_trace: bool,
    with_seo: bool,
    test: bool = True,
    test_times: int = 0,
    thr_granularity: int = 200,
) -> dict[ParamSet, tuple[ExpTrainResult, ExpTestResult | None]]:
    thr_candidates: list[float] = np.linspace(0, 1, thr_granularity).tolist()

    # pre-calculate error and confmat by threshold
    err_by_thr_idx: list[float] = [0 for _ in thr_candidates]
    confmat_by_thr_idx: list[tuple[float, float, float, float]] = [
        (0, 0, 0, 0) for _ in thr_candidates
    ]

    for thr_idx, thr in enumerate(thr_candidates):
        _, confmat = classifier.predict_train(thr)
        tn, fp, fn, tp = confmat.astype(float)
        err_by_thr_idx[thr_idx] = (fp + fn) / (tn + fp + fn + tp)
        confmat_by_thr_idx[thr_idx] = (tn, fp, fn, tp)

    # pre-calculate total/group rfp/rfn by threshold
    total_rfp_cache_by_thr_idx: NDArray[np.float_] | None = None
    total_rfn_cache_by_thr_idx: NDArray[np.float_] | None = None
    group_rfp_cache_by_thr_idx: NDArray[np.float_] | None = None
    group_rfn_cache_by_thr_idx: NDArray[np.float_] | None = None
    group_size_ratio: NDArray[np.float_] | None = None

    if with_seo:
        total_rfp_cache_by_thr_idx = np.array(
            [fp / (tn + fp + fn + tp) for tn, fp, fn, tp in confmat_by_thr_idx]
        )
        total_rfn_cache_by_thr_idx = np.array(
            [fn / (tn + fp + fn + tp) for tn, fp, fn, tp in confmat_by_thr_idx]
        )

        group_rfp_cache_by_thr_idx = np.zeros(
            (classifier.n_groups, len(thr_candidates)), dtype=np.float_
        )
        group_rfn_cache_by_thr_idx = np.zeros(
            (classifier.n_groups, len(thr_candidates)), dtype=np.float_
        )
        group_size_ratio = np.zeros(classifier.n_groups, dtype=np.float_)
        group_n_sum = 0
        for group_idx, (_, group_indices) in enumerate(
            classifier.iter_train_groups()
        ):
            group_size_ratio[group_idx] = len(group_indices)
            group_n_sum += len(group_indices)
            for thr_idx, thr in enumerate(thr_candidates):
                _, confmat = classifier.predict_train(
                    thr, indices=group_indices
                )
                tn, fp, fn, tp = confmat.astype(float)
                group_rfp_cache_by_thr_idx[group_idx, thr_idx] = fp / (
                    tn + fp + fn + tp
                )
                group_rfn_cache_by_thr_idx[group_idx, thr_idx] = fn / (
                    tn + fp + fn + tp
                )
        group_size_ratio /= group_n_sum

    # pre-calculate ge/err baseline index
    baseline_idx = np.argmin(err_by_thr_idx)

    # pre-calculate generalized entropy by alpha/c/a
    ge_cache = Cache[float, list[float]]()

    for alpha in set(param_dict["alpha"]):
        for c in set(param_dict["c"]):
            for a in set(param_dict["a"]):
                ge_list: list[float] = [0 for _ in thr_candidates]

                for thr_idx, thr in enumerate(thr_candidates):
                    tn, fp, fn, tp = confmat_by_thr_idx[thr_idx]
                    ge_list[thr_idx] = ge_confmat(alpha, c, a, tn, fp, fn, tp)

                ge_cache.set(ge_list, alpha=alpha, c=c, a=a)

    # build GEFairSolver and save threading arguments with memory consumption
    lib_path = GEFairSolver.compile_gefair(trace_hypi_t=with_trace)
    solver = GEFairSolver(lib_path)  # to test build and get memory usage

    threading_mem_usage: list[float] = []
    threading_args = []
    for param_set in get_param_sets(param_dict):
        threading_mem_usage.append(
            solver.predict_memory_usage(
                len(thr_candidates),
                param_set.lambda_max,
                param_set.nu,
                param_set.alpha,
                param_set.gamma,
                param_set.c,
                param_set.a,
            )
        )
        threading_args.append((param_set,))

    del solver

    # define runner function
    def runner(ps: ParamSet) -> tuple[ParamSet, GEFairResult]:
        return ps, GEFairSolver(lib_path).solve_gefair(
            thr_candidates=thr_candidates,
            I_alpha_cache=ge_cache.get(alpha=ps.alpha, c=ps.c, a=ps.a),
            err_cache=err_by_thr_idx,
            lambda_max=ps.lambda_max,
            nu=ps.nu,
            alpha=ps.alpha,
            gamma=ps.gamma,
            c=ps.c,
            a=ps.a,
        )

    # execute in parallel
    pool_size = min(
        max(1, psutil.cpu_count(logical=False) - 1),
        len(threading_args),
    )
    pool = ThreadPool(processes=pool_size)
    results: dict[ParamSet, tuple[ExpTrainResult, ExpTestResult | None]] = {}
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
        for runner_retval in pool.starmap(runner, args_to_be_used):
            ps: ParamSet = runner_retval[0]
            gefair_result: GEFairResult = runner_retval[1]

            # collect training results
            ge_bar_baseline = ge_cache.get(alpha=ps.alpha, c=ps.c, a=ps.a)[
                baseline_idx
            ]
            err_bar_baseline = err_by_thr_idx[baseline_idx]

            ge_bar_trace: NDArray[np.float_] | None = None
            err_bar_trace: NDArray[np.float_] | None = None
            if gefair_result.hypi_t is not None:
                ge_bar_trace = get_time_averaged_trace(
                    gefair_result.hypi_t,
                    np.array(ge_cache.get(alpha=ps.alpha, c=ps.c, a=ps.a)),
                )
                err_bar_trace = get_time_averaged_trace(
                    gefair_result.hypi_t, np.array(err_by_thr_idx)
                )

            mseo_trace: NDArray[np.float_] | None = None
            aseo_trace: NDArray[np.float_] | None = None
            if with_seo and gefair_result.hypi_t is not None:
                assert total_rfp_cache_by_thr_idx is not None
                assert total_rfn_cache_by_thr_idx is not None
                assert group_rfp_cache_by_thr_idx is not None
                assert group_rfn_cache_by_thr_idx is not None
                assert group_size_ratio is not None
                mseo_trace, aseo_trace = get_seo_trace(
                    gefair_result.hypi_t,
                    (total_rfp_cache_by_thr_idx, total_rfn_cache_by_thr_idx),
                    (group_rfp_cache_by_thr_idx, group_rfn_cache_by_thr_idx),
                    group_size_ratio,
                )

            train_result = ExpTrainResult(
                ge_baseline=ge_bar_baseline,
                err_baseline=err_bar_baseline,
                ge_bar_trace=ge_bar_trace,
                err_bar_trace=err_bar_trace,
                mseo_trace=mseo_trace,
                aseo_trace=aseo_trace,
            )

            # collect testing results
            test_result: ExpTestResult | None = None
            if test:
                hyps, probs = zip(
                    *(
                        {
                            hyp: count / gefair_result.T
                            for hyp, count in gefair_result.hyp_stat.items()
                        }
                    ).items()
                )
                probs_np = np.array(probs)

                test_confmats = [
                    classifier.predict_test(hyp)[1] for hyp in hyps
                ]
                test_ges = np.array(
                    [
                        ge_confmat(ps.alpha, ps.c, ps.a, *confmat)
                        for confmat in test_confmats
                    ]
                )
                test_errs = np.array(
                    [
                        (fp + fn) / (tn + fp + fn + tp)
                        for tn, fp, fn, tp in test_confmats
                    ]
                )

                test_ge_mean, test_ge_std = get_mean_std(
                    test_ges, probs_np, times=test_times
                )
                test_err_mean, test_err_std = get_mean_std(
                    test_errs, probs_np, times=test_times
                )

                test_result = ExpTestResult(
                    ge=test_ge_mean,
                    ge_std=test_ge_std,
                    err=test_err_mean,
                    err_std=test_err_std,
                    mseo=None,
                    mseo_std=None,
                    aseo=None,
                    aseo_std=None,
                )

            # save results
            results[ps] = train_result, test_result

            del gefair_result

        # finish
        gc.collect()

        if len(threading_args) == 0:
            break

    pool.close()
    return results
