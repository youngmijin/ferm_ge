import argparse
import gc
import importlib
import inspect
import os
import pickle
import time
import types
from collections import defaultdict

import matplotlib.pyplot as plt
import numba
import numpy as np
import rich.traceback
import yaml
from rich import print

from data import Dataset
from ferm_ge import BinaryLogisticClassification, get_param_sets, run_exp
from plotting import (
    DEFAULT_FIGSIZE,
    plot_training_traces_by_c,
    plot_valid_results_by_gamma_c,
    save_fig,
)

rich.traceback.install(show_locals=True, suppress=[numba])


def make_param_readable(param: list[float]) -> str | float:
    if len(param) == 1:
        return param[0]
    elif len(param) == 0:
        return "(empty list)"
    elif len(param) <= 5:
        return ", ".join([str(p) for p in param])
    else:
        return f"{min(param)} to {max(param)} (total {len(param)} steps)"


def versatile_float(s: str) -> list[float]:
    if s.startswith("np."):
        return [float(i) for i in np.ndarray.tolist(eval(s).astype(float))]
    if s.startswith("range("):
        return [float(i) for i in eval(s)]
    return [float(s)]


def parse_plotting_rules(
    strs: list[str], default: str | None = None
) -> defaultdict[str, str | None]:
    optdict = defaultdict[str, str | None](lambda: default)
    for s in strs:
        assert ":" in s, "plotting parameter must be in the form of 'key:value'"
        key, value = s.split(":")
        assert (
            "=" in key
        ), "plotting parameter must be in the form of 'key=value:value'"
        optdict[key] = value
    return optdict


def main(
    # algorithm options
    lambda_max: list[list[float]],
    nu: list[list[float]],
    alpha: list[list[float]],
    gamma: list[list[float]],
    c: list[list[float]],
    a: list[list[float]],
    # run options
    study_type: str,
    dataset: str,
    dataset_url: str | None,
    blc_max_iter: int,
    calc_seo: bool,
    seo_components: list[str],
    thr_granularity: int,
    valid_times: int,
    no_threading: bool,
    # plotting options
    metrics: list[list[str]],
    metrics_right: list[str],
    figsize: tuple[float, float],
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    coloring_rules: defaultdict[str, str | None],
    styling_rules: defaultdict[str, str | None],
    confidence_band: float,
    no_baseline: bool,
    use_tex: bool,
    # output options
    output_dir: str,
    save_pkl: bool,
):
    os.makedirs(output_dir, exist_ok=True)
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # analyze given parameters and create parameter groups for experiment units
    len_params = [
        1,
        len(lambda_max),
        len(nu),
        len(alpha),
        len(gamma),
        len(c),
        len(a),
    ]
    assert (
        len(set(len_params)) < 3  # ♡
    ), "all parameters' length must be 1 or have the same length other than 1"

    param_dicts = [
        {
            "lambda_max": lambda_max[0]
            if len(lambda_max) == 1
            else lambda_max[unit_idx],
            "nu": nu[0] if len(nu) == 1 else nu[unit_idx],
            "alpha": alpha[0] if len(alpha) == 1 else alpha[unit_idx],
            "gamma": gamma[0] if len(gamma) == 1 else gamma[unit_idx],
            "c": c[0] if len(c) == 1 else c[unit_idx],
            "a": a[0] if len(a) == 1 else a[unit_idx],
        }
        for unit_idx in range(max(len_params))
    ]

    # load dataset
    data_class = None
    for v in importlib.import_module(f"data.{dataset}").__dict__.values():
        if (
            inspect.isclass(v)
            and (not inspect.isabstract(v))
            and (type(v) != types.GenericAlias)
            and (v is not Dataset)
            and issubclass(v, Dataset)
        ):
            data_class = v
            break
    if data_class is None:
        raise ValueError(f"invalid dataset: {dataset}")
    data = data_class()
    data.download(remote_url=dataset_url)
    data.load()
    print("dataset:", data.name)

    # pre-train classifier
    classifier = BinaryLogisticClassification(max_iter=blc_max_iter)
    classifier.train(*data.train_data)
    classifier.valid(*data.valid_data)
    classifier.set_group(
        data.train_group_indices,
        data.valid_group_indices,
    )

    print()

    # set up unit configs
    UNIT_TRAIN_LOG_TRACE: bool
    UNIT_TRAIN_CALC_SEO: bool
    UNIT_VALID_DO: bool
    UNIT_VALID_CALC_SEO: bool
    UNIT_METRICS_TRACE: list[list[str]]
    UNIT_METRICS_BY_GAMMA: list[list[str]]

    if study_type == "convergence":
        UNIT_TRAIN_LOG_TRACE = True
        UNIT_TRAIN_CALC_SEO = calc_seo
        UNIT_VALID_DO = False
        UNIT_VALID_CALC_SEO = False
        UNIT_METRICS_TRACE = metrics
        UNIT_METRICS_BY_GAMMA = []
    elif study_type == "varying_gamma":
        UNIT_TRAIN_LOG_TRACE = False
        UNIT_TRAIN_CALC_SEO = False
        UNIT_VALID_DO = True
        UNIT_VALID_CALC_SEO = calc_seo
        UNIT_METRICS_TRACE = []
        UNIT_METRICS_BY_GAMMA = metrics
    else:
        raise ValueError(f"invalid study_type: {study_type}")

    # run experiments
    for unit_idx, unit_param_dict in enumerate(param_dicts):
        print(f"<🧩 experiment unit [ {unit_idx + 1} / {len(param_dicts)} ]>")
        print("param_dict:", unit_param_dict)
        print("param_sets:", len(get_param_sets(unit_param_dict)))
        fname_prefix = f"{run_name}_{data.name}_{unit_idx}"

        # save parameter dictionary
        unit_param_dict_readable: dict[str, str | float] = {}
        unit_param_dict_readable["dataset"] = data.name
        unit_param_dict_readable["thr_granularity"] = thr_granularity
        if len(metrics_right) > 0:
            all_metrics = [m for ms in metrics for m in ms]
            unit_param_dict_readable["plot_left_y_axis"] = ", ".join(
                [m for m in all_metrics if m not in metrics_right]
            )
            unit_param_dict_readable["plot_right_y_axis"] = ", ".join(
                [m for m in metrics_right if m in all_metrics]
            )
        for k, v in unit_param_dict.items():
            unit_param_dict_readable[k] = make_param_readable(v)
        with open(os.path.join(output_dir, f"{fname_prefix}.yaml"), "w") as sf:
            yaml.dump(unit_param_dict_readable, sf)

        # run experiments
        print("experiment: ", flush=True, end="")
        start_time = time.time()
        results = run_exp(
            classifier,
            unit_param_dict,
            keep_trace=UNIT_TRAIN_LOG_TRACE,
            calc_train_seo=UNIT_TRAIN_CALC_SEO,
            calc_valid_seo=UNIT_VALID_CALC_SEO,
            seo_components=seo_components,
            include_valid=UNIT_VALID_DO,
            valid_times=valid_times,
            thr_granularity=thr_granularity,
            no_threading=no_threading,
        )
        print(f"done in {time.time() - start_time:.2f} sec", flush=True, end="")
        if save_pkl:
            results_pkl_path = os.path.join(
                output_dir, f"{fname_prefix}_results.pkl"
            )
            with open(results_pkl_path, "wb") as bf:
                pickle.dump(results, bf)
            print(f", results saved to {results_pkl_path}", flush=True)
        else:
            print(flush=True)

        train_results = {ps: result[0] for ps, result in results.items()}
        valid_results = {ps: result[1] for ps, result in results.items()}

        # plot results
        print("plotting:", flush=True)

        for m in UNIT_METRICS_TRACE:
            fig = plot_training_traces_by_c(
                train_results,
                metrics=m,
                metrics_right=metrics_right,
                figsize=figsize,
                coloring_rules=coloring_rules,
                styling_rules=styling_rules,
                use_tex=use_tex,
            )
            m_trace_pdf_path = os.path.join(
                output_dir,
                f"{fname_prefix}_{'-'.join(m)}_trace.pdf",
            )
            save_fig(fig, m_trace_pdf_path)
            print(f"  ├─ {m} trace saved to {m_trace_pdf_path}")
            if xlim is not None or ylim is not None:
                m_trace_magni_pdf_path = os.path.join(
                    output_dir,
                    f"{fname_prefix}_{'-'.join(m)}_trace_zoom.pdf",
                )
                save_fig(fig, m_trace_magni_pdf_path, xlim, ylim)
                print(f"  ├─ {m} trace saved to {m_trace_magni_pdf_path}")
            fig.clear()
            plt.close(fig)

        for m in UNIT_METRICS_BY_GAMMA:
            fig = plot_valid_results_by_gamma_c(
                valid_results,  # type: ignore
                metrics=m,
                baselines=(
                    None
                    if no_baseline
                    else {
                        ps: {
                            "ge": result.ge_baseline,
                            "err": result.err_baseline,
                        }
                        for ps, result in train_results.items()
                    }
                ),
                metrics_right=metrics_right,
                figsize=figsize,
                coloring_rules=coloring_rules,
                styling_rules=styling_rules,
                confidence_band=confidence_band,
                use_tex=use_tex,
            )
            m_by_gamma_pdf_path = os.path.join(
                output_dir,
                f"{fname_prefix}_{'-'.join(m)}_by_gamma.pdf",
            )
            save_fig(fig, m_by_gamma_pdf_path)
            fig.clear()
            plt.close(fig)
            print(f"  ├─ {m} by gamma saved to {m_by_gamma_pdf_path}")

        print("  └─ (drawing done)", flush=True)

        # clean up
        del results
        gc.collect()
        print(flush=True)

    print("🪄 all done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no_threading",
        action="store_true",
        help="disable threading if set",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        metavar="PATH",
        help="path to the output directory",
    )
    parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="save results as pickle file (this may take a lot of disk space)",
    )

    algopt = parser.add_argument_group(
        "algorithm options", "refer to the paper for details"
    )
    algopt.add_argument(
        "--lambda_max",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    algopt.add_argument(
        "--nu",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    algopt.add_argument(
        "--alpha",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    algopt.add_argument(
        "--gamma",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    algopt.add_argument(
        "--c",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    algopt.add_argument(
        "--a",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )

    expopt = parser.add_argument_group("experiment options")
    expopt.add_argument(
        "--study_type",
        type=str,
        required=True,
        choices=[
            "convergence",
            "varying_gamma",
        ],
        help="study type to run",
    )
    expopt.add_argument(
        "--dataset",
        type=str,
        required=True,
        metavar="NAME",
        help="dataset name; must be same with the python file name in ./data",
    )
    expopt.add_argument(
        "--dataset_url",
        type=str,
        default=None,
        metavar="URL",
        help="dataset url to download; use if hard-coded dataset url is broken",
    )
    expopt.add_argument(
        "--blc_max_iter",
        type=int,
        default=1000,
        metavar="N",
        help="maximum number of iterations for binary logistic classifier",
    )
    expopt.add_argument(
        "--calc_seo",
        action="store_true",
        help="calculate SEO and V values if set",
    )
    expopt.add_argument(
        "--seo_components",
        type=str,
        default="fp+fn",
        choices=["fp+fn", "fp", "fn"],
        help="SEO components to calculate",
    )
    expopt.add_argument(
        "--thr_granularity",
        type=int,
        default=200,
        metavar="N",
        help="threshold granularity; same as hyper-parameter granularity",
    )
    expopt.add_argument(
        "--valid_times",
        type=int,
        default=0,
        metavar="N",
        help="times to repeat stochastic test; set 0 for deterministic test",
    )

    plotopt = parser.add_argument_group("plotting options")
    plotopt.add_argument(
        "--metrics",
        action="append",
        type=str,
        nargs="*",
        required=True,
        choices=["ge_bar", "ge", "err_bar", "err", "mseo", "aseo", "v"],
        help="metrics to plot",
    )
    plotopt.add_argument(
        "--metrics_right",
        type=str,
        nargs="*",
        choices=["ge_bar", "ge", "err_bar", "err", "mseo", "aseo", "v"],
        default=[],
        help="metrics to plot on the right y-axis (applied for all plots)",
    )
    plotopt.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=DEFAULT_FIGSIZE,
        metavar=("WIDTH", "HEIGHT"),
        help="figure size for plots",
    )
    plotopt.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LEFT", "RIGHT"),
        help="xlim range; only applied for 'convergence' studies",
    )
    plotopt.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("BOTTOM", "TOP"),
        help="ylim range; only applied for 'convergence' studies",
    )
    plotopt.add_argument(
        "--coloring_rules",
        type=str,
        nargs="+",
        default=None,
        metavar="RULE",
        help="line colors, by algorithm options and metrics (e.g. c=0.9:red)",
    )
    plotopt.add_argument(
        "--styling_rules",
        type=str,
        nargs="+",
        default=None,
        metavar="RULE",
        help="line styles, by algorithm options and metrics (e.g. m=err:solid)",
    )
    plotopt.add_argument(
        "--confidence_band",
        type=float,
        default=0.0,
        metavar="P",
        help="confidence (P%%) band; only applied for 'varying_gamma' studies",
    )
    plotopt.add_argument(
        "--no_baseline",
        action="store_true",
        help="remove dashed baseline; only applied for 'varying_gamma' studies",
    )
    plotopt.add_argument(
        "--use_tex",
        action="store_true",
        help="use tex for rendering text",
    )

    args = parser.parse_args()

    # NOTE: in algorithm options, list is overlapped three times to allow for
    #       multiple values
    #         - 1st list = list from action="append"
    #                      experiment units will be created for this level
    #         - 2nd list = list from nargs="+"
    #         - 3rd list = list from versatile_float
    #       so 2nd and 3rd lists will be flattened into one list
    #       (e.g. [[[1, 2], [3]], [[5], [7, 8]]] -> [[1, 2, 3], [5, 7, 8]])
    args.lambda_max = [[f for s in l for f in s] for l in args.lambda_max]
    args.nu = [[f for s in l for f in s] for l in args.nu]
    args.gamma = [[f for s in l for f in s] for l in args.gamma]
    args.alpha = [[f for s in l for f in s] for l in args.alpha]
    args.c = [[f for s in l for f in s] for l in args.c]
    args.a = [[f for s in l for f in s] for l in args.a]

    args.seo_components = args.seo_components.split("+")

    args.figsize = tuple(args.figsize)
    args.xlim = tuple(args.xlim) if args.xlim is not None else None
    args.ylim = tuple(args.ylim) if args.ylim is not None else None
    args.coloring_rules = parse_plotting_rules(args.coloring_rules or [])
    args.styling_rules = parse_plotting_rules(args.styling_rules or [])

    print("args:", args)
    main(**args.__dict__)
