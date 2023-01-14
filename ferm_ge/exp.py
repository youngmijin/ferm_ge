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
from .exp_utils import Cache, get_mean_std, get_time_averaged_trace
from .task_blc import BinaryLogisticClassification


@dataclass
class ExpTestResult:
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
    calc_test_seo: bool = False,
    seo_components: list[str] = ["fp", "fn"],
    include_test: bool = True,
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

    # pre-calculate seo by threshold if needed
    mseo_train_by_thr_idx: NDArray[np.float_] | None = None
    aseo_train_by_thr_idx: NDArray[np.float_] | None = None

    group_cnt = len(classifier.group_names)
    if calc_train_seo:
        total_rfp_train = np.array(
            [fp / (tn + fp + fn + tp) for tn, fp, fn, tp in confmat_by_thr_idx]
        )
        total_rfn_train = np.array(
            [fn / (tn + fp + fn + tp) for tn, fp, fn, tp in confmat_by_thr_idx]
        )

        group_rfp_train = np.zeros(
            (group_cnt, thr_granularity), dtype=np.float_
        )
        group_rfn_train = np.zeros(
            (group_cnt, thr_granularity), dtype=np.float_
        )
        group_size_train: NDArray[np.int_] = np.zeros(group_cnt, dtype=np.int_)
        for group_idx, group_name in enumerate(classifier.group_names):
            group_n_train = 0
            for thr_idx, thr in enumerate(thr_candidates):
                _, confmat = classifier.predict_train(thr, group=group_name)
                tn, fp, fn, tp = confmat.astype(float)
                group_n_train = int(tn + fp + fn + tp)
                group_rfp_train[group_idx, thr_idx] = fp / group_n_train
                group_rfn_train[group_idx, thr_idx] = fn / group_n_train
            group_size_train[group_idx] = group_n_train

        mseo_train_by_thr_idx = np.zeros(thr_granularity, dtype=np.float_)
        aseo_train_by_thr_idx = np.zeros(thr_granularity, dtype=np.float_)
        for thr_idx in range(thr_granularity):
            (
                mseo_train_by_thr_idx[thr_idx],
                aseo_train_by_thr_idx[thr_idx],
            ) = calc_seo(
                total_rfp_train[thr_idx] if "fp" in seo_components else None,
                total_rfn_train[thr_idx] if "fn" in seo_components else None,
                group_rfp_train[:, thr_idx] if "fp" in seo_components else None,
                group_rfn_train[:, thr_idx] if "fn" in seo_components else None,
                group_size_train,
            )

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
                    ge_list[thr_idx] = calc_ge_confmat(
                        alpha, c, a, tn, fp, fn, tp
                    )

                ge_cache.set(ge_list, alpha=alpha, c=c, a=a)

    # build GEFairSolver and save threading arguments with memory consumption
    lib_path = GEFairSolver.compile_gefair(trace_hypi_t=keep_trace)
    solver = GEFairSolver(lib_path)  # to test build and get memory usage

    threading_mem_usage: list[float] = []
    threading_args = []
    for param_set in get_param_sets(param_dict):
        mem_gefair = solver.predict_memory_usage(
            len(thr_candidates),
            param_set.lambda_max,
            param_set.nu,
            param_set.alpha,
            param_set.gamma,
            param_set.c,
            param_set.a,
        )
        threading_mem_usage.append(mem_gefair)
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

            # collect training results - 1 (generalized entropy & error trace)
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

            # collect training results - 2 (mseo & aseo trace)
            mseo_trace: NDArray[np.float_] | None = None
            aseo_trace: NDArray[np.float_] | None = None
            if calc_train_seo and gefair_result.hypi_t is not None:
                assert mseo_train_by_thr_idx is not None
                assert aseo_train_by_thr_idx is not None
                mseo_trace = mseo_train_by_thr_idx[gefair_result.hypi_t]
                aseo_trace = aseo_train_by_thr_idx[gefair_result.hypi_t]

            train_result = ExpTrainResult(
                ge_baseline=ge_bar_baseline,
                err_baseline=err_bar_baseline,
                ge_bar_trace=ge_bar_trace,
                err_bar_trace=err_bar_trace,
                mseo_trace=mseo_trace,
                aseo_trace=aseo_trace,
            )

            # collect testing results if needed
            test_result: ExpTestResult | None = None
            if include_test:
                # collect testing results - 1 (generalized entropy & error)
                result_hypis = gefair_result.hypi_stat.keys()
                result_hypi_probs = (
                    np.array(
                        list(gefair_result.hypi_stat.values()),
                        dtype=np.float_,
                    )
                    / gefair_result.T
                )

                test_confmats = [
                    classifier.predict_test(thr_candidates[hypi])[1]
                    for hypi in result_hypis
                ]
                test_ges = np.array(
                    [
                        calc_ge_confmat(ps.alpha, ps.c, ps.a, *confmat)
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
                    test_ges, result_hypi_probs, times=test_times
                )
                test_err_mean, test_err_std = get_mean_std(
                    test_errs, result_hypi_probs, times=test_times
                )

                # collect testing results - 3 (v & mseo & aseo)
                test_v_mean: float | None = None
                test_v_std: float | None = None
                test_mseo_mean: float | None = None
                test_mseo_std: float | None = None
                test_aseo_mean: float | None = None
                test_aseo_std: float | None = None

                if calc_test_seo:
                    group_tp_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )
                    group_fp_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )
                    group_fn_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )
                    group_tn_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )

                    group_rfp_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )
                    group_rfn_test = np.zeros(
                        (group_cnt, len(result_hypis)), dtype=np.float_
                    )
                    group_size_test: NDArray[np.int_] = np.zeros(
                        (group_cnt,), dtype=np.int_
                    )
                    for group_idx, group_name in enumerate(
                        classifier.group_names
                    ):
                        group_n_test = 0
                        for hypi_idx, hypi in enumerate(result_hypis):
                            thr = thr_candidates[hypi]
                            _, confmat = classifier.predict_test(
                                thr, group=group_name
                            )
                            tn, fp, fn, tp = confmat.astype(float)
                            group_n_test = int(tn + fp + fn + tp)
                            group_tp_test[group_idx, hypi_idx] = tp
                            group_fp_test[group_idx, hypi_idx] = fp
                            group_fn_test[group_idx, hypi_idx] = fn
                            group_tn_test[group_idx, hypi_idx] = tn
                            group_rfp_test[group_idx, hypi_idx] = (
                                fp / group_n_test
                            )
                            group_rfn_test[group_idx, hypi_idx] = (
                                fn / group_n_test
                            )
                        group_size_test[group_idx] = group_n_test

                    v_test = np.zeros_like(result_hypi_probs)
                    for i in range(len(result_hypis)):
                        v_test[i] = calc_ge_v(
                            ps.alpha,
                            ps.c,
                            ps.a,
                            group_tp_test[:, i],
                            group_fp_test[:, i],
                            group_fn_test[:, i],
                            group_tn_test[:, i],
                        )

                    test_v_mean, test_v_std = get_mean_std(
                        v_test, result_hypi_probs, times=test_times
                    )

                    mseo_test = np.zeros_like(result_hypi_probs)
                    aseo_test = np.zeros_like(result_hypi_probs)
                    for i in range(len(result_hypis)):
                        tn, fp, fn, tp = test_confmats[i]
                        total_rfp_test: float = fp / (tn + fp + fn + tp)
                        total_rfn_test: float = fn / (tn + fp + fn + tp)

                        mseo_test[i], aseo_test[i] = calc_seo(
                            total_rfp_test if "fp" in seo_components else None,
                            total_rfn_test if "fn" in seo_components else None,
                            group_rfp_test[:, i]
                            if "fp" in seo_components
                            else None,
                            group_rfn_test[:, i]
                            if "fn" in seo_components
                            else None,
                            group_size_test,
                        )

                    test_mseo_mean, test_mseo_std = get_mean_std(
                        mseo_test, result_hypi_probs, times=test_times
                    )
                    test_aseo_mean, test_aseo_std = get_mean_std(
                        aseo_test, result_hypi_probs, times=test_times
                    )

                test_result = ExpTestResult(
                    ge=test_ge_mean,
                    ge_std=test_ge_std,
                    err=test_err_mean,
                    err_std=test_err_std,
                    v=test_v_mean,
                    v_std=test_v_std,
                    mseo=test_mseo_mean,
                    mseo_std=test_mseo_std,
                    aseo=test_aseo_mean,
                    aseo_std=test_aseo_std,
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
