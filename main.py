import argparse
import gc
import importlib
import inspect
import os
import pickle
import time
import types
from typing import Any

import matplotlib.pyplot as plt
import numpy as np  # required for versatile_float
import yaml
from numpy.typing import NDArray

from data import Dataset
from ferm_ge import (
    DEFAULT_COLORS,
    DEFAULT_FIGSIZE,
    DEFAULT_STYLES,
    BinaryLogisticClassification,
    get_param_sets,
    plot_test_results_by_gamma,
    plot_training_traces,
    run_exp,
)


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
    if "np." in s:
        npl: NDArray[Any] = eval(s)
        fl: list[float] = npl.astype(float).tolist()
        return fl
    if "range(" in s:
        fl = [float(i) for i in eval(s)]
        return fl
    return [float(s)]


def flatten_list(l: list[list[float]]) -> list[float]:
    return [item for sublist in l for item in sublist]


def main(args: argparse.Namespace):
    print("args:", args)

    # analyze given parameters and create parameter groups
    assert (
        len(
            set(
                [
                    1,
                    len(args.lambda_max),
                    len(args.nu),
                    len(args.alpha),
                    len(args.gamma),
                    len(args.c),
                    len(args.a),
                ]
            )
        )
        < 3  # ♡
    ), "all parameters' length must be 1 or have the same length other than 1."

    def get_arg_list(name: str, group_idx: int) -> list[float]:
        if len(args.__dict__[name]) == 1:
            return flatten_list(args.__dict__[name][0])
        return flatten_list(args.__dict__[name][group_idx])

    param_dicts: list[dict[str, list[float]]] = []
    for group_idx in range(len(args.alpha)):
        param_dicts.append(
            {
                "lambda_max": get_arg_list("lambda_max", group_idx),
                "nu": get_arg_list("nu", group_idx),
                "alpha": get_arg_list("alpha", group_idx),
                "gamma": get_arg_list("gamma", group_idx),
                "c": get_arg_list("c", group_idx),
                "a": get_arg_list("a", group_idx),
            }
        )

    # load dataset
    dataset_class = None
    for v in importlib.import_module(f"data.{args.dataset}").__dict__.values():
        if (
            inspect.isclass(v)
            and (not inspect.isabstract(v))
            and (type(v) != types.GenericAlias)
            and (v is not Dataset)
            and issubclass(v, Dataset)
        ):
            dataset_class = v
            break
    if dataset_class is None:
        raise ValueError(f"invalid dataset: {args.dataset}")
    dataset: Dataset = dataset_class()
    print("dataset:", dataset.name)

    # pre-train classifier
    classifier = BinaryLogisticClassification()
    classifier.train(*dataset.get_train_data())
    classifier.test(*dataset.get_test_data())
    classifier.set_group(
        dataset.get_train_group_indices(),
        dataset.get_test_group_indices(),
    )

    print()

    # run experiments
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    for group_idx, group_param_dict in enumerate(param_dicts):
        print(f"<group [ {group_idx + 1} / {len(param_dicts)} ]>")
        print("param_dict:", group_param_dict)
        print("param_sets:", len(get_param_sets(group_param_dict)))

        OPT_WITH_TRACE = False
        OPT_DO_TEST = False
        OPT_PLOT_BY_GAMMA = False
        OPT_CALC_SEO = False

        assert args.study_type in ["convergence", "varying_gamma"]
        if args.study_type == "convergence":
            OPT_WITH_TRACE = True

        if (
            args.study_type == "varying_gamma"
            and len(group_param_dict["gamma"]) > 1
        ):
            OPT_DO_TEST = True
            OPT_PLOT_BY_GAMMA = True

        if args.calc_seo:
            OPT_CALC_SEO = True

        # save parameter dictionary
        group_param_dict_readable: dict[str, str | float] = {}
        group_param_dict_readable["dataset"] = dataset.name
        group_param_dict_readable["thr_granularity"] = args.thr_granularity
        for k, v in group_param_dict.items():
            group_param_dict_readable[k] = make_param_readable(v)
        with open(
            os.path.join(
                args.output_dir,
                f"{run_name}_{dataset.name}_{group_idx}.yaml",
            ),
            "w",
        ) as strf:
            yaml.dump(group_param_dict_readable, strf)

        # run experiments
        print("experiment: ", flush=True, end="")
        start_time = time.time()
        results = run_exp(
            classifier,
            group_param_dict,
            with_trace=OPT_WITH_TRACE,
            with_seo=OPT_CALC_SEO,
            test=OPT_DO_TEST,
            test_times=args.test_times,
            thr_granularity=args.thr_granularity,
        )
        print(f"done in {time.time() - start_time:.2f} sec", flush=True, end="")
        if args.save_pkl:
            results_pkl_path = os.path.join(
                args.output_dir,
                f"{run_name}_{dataset.name}_{group_idx}_results.pkl",
            )
            with open(results_pkl_path, "wb") as binf:
                pickle.dump(results, binf)
            print(f", results saved to {results_pkl_path}", flush=True)
        else:
            print(flush=True)

        train_results = {ps: result[0] for ps, result in results.items()}
        test_results = {ps: result[1] for ps, result in results.items()}

        # plot results
        print("plotting:", flush=True)
        if OPT_WITH_TRACE:
            metrics = ["ge_bar", "err_bar"]
            if OPT_CALC_SEO:
                metrics += ["mseo", "aseo"]
            for m in metrics:
                fig = plot_training_traces(
                    train_results,
                    metric_names=m,
                    figsize=args.figsize,
                    colors=args.colors,
                    styles=args.styles,
                )
                m_trace_pdf_path = os.path.join(
                    args.output_dir,
                    f"{run_name}_{dataset.name}_{group_idx}_{m}.pdf",
                )
                fig.savefig(
                    m_trace_pdf_path,
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=600,
                )
                print(f"  {m} trace saved to {m_trace_pdf_path}")

                if args.magnify is not None:
                    fig.gca().set_xlim(*args.magnify)
                    m_trace_jitter_pdf_path = os.path.join(
                        args.output_dir,
                        f"{run_name}_{dataset.name}_{group_idx}_{m}_jitter.pdf",
                    )
                    fig.savefig(
                        m_trace_jitter_pdf_path,
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=600,
                    )
                    print(f"  {m} trace saved to {m_trace_jitter_pdf_path}")

                fig.clear()
                plt.close(fig)

        if OPT_PLOT_BY_GAMMA:
            for x in test_results.values():
                assert x is not None, "test results not available"
            metrics = ["ge", "err"]
            if OPT_CALC_SEO:
                metrics += ["mseo", "aseo"]
            for m in metrics:
                m_by_gamma_pdf_path = os.path.join(
                    args.output_dir,
                    f"{run_name}_{dataset.name}_{group_idx}_{m}_by_gamma.pdf",
                )
                fig = plot_test_results_by_gamma(
                    test_results,  # type: ignore
                    metric_names=m,
                    baselines=(
                        {
                            ps: {m: getattr(result, f"{m}_baseline")}
                            for ps, result in train_results.items()
                        }
                    )
                    if m in ["ge", "err"]
                    else None,
                    fname=m_by_gamma_pdf_path,
                    figsize=args.figsize,
                    colors=args.colors,
                    styles=args.styles,
                )
                fig.clear()
                plt.close(fig)
                print(f"  {m} by gamma saved to {m_by_gamma_pdf_path}")

        # clean up
        del results
        gc.collect()
        print(flush=True)

    print("🪄 all done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lambda_max",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--nu",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--alpha",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--gamma",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--c",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--a",
        action="append",
        type=versatile_float,
        nargs="+",
        required=True,
    )

    parser.add_argument("--study_type", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--calc_seo", action="store_true")
    parser.add_argument("--thr_granularity", type=int, default=200)
    parser.add_argument("--test_times", type=int, default=0)

    parser.add_argument("--magnify", type=float, nargs=2, default=None)
    parser.add_argument("--colors", type=str, nargs="+", default=DEFAULT_COLORS)
    parser.add_argument("--styles", type=str, nargs="+", default=DEFAULT_STYLES)
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=DEFAULT_FIGSIZE
    )

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--save_pkl", action="store_true")

    args = parser.parse_args()
    main(args)
