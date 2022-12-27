import argparse
import gc
import importlib
import inspect
import os
import pickle
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib._color_data import BASE_COLORS, CSS4_COLORS

from data import Preset
from ferm_ge import (
    Experiment,
    frozenkey_to_paramdict,
    plot_convergence,
    plot_metrics,
    plotting_default_colors,
    plotting_default_figsize,
)


def versatile_float(s: str) -> List[float]:
    if "np." in s:
        npl: np.ndarray = eval(s)
        fl: List[float] = npl.astype(float).tolist()
        return fl
    if "range" in s:
        fl = [float(i) for i in eval(s)]
        return fl
    return [float(s)]


def flatten_list(l: List[List[float]]) -> List[float]:
    return [item for sublist in l for item in sublist]


def main(args):
    # Check if colors are valid
    for color in args.colors:
        if color not in BASE_COLORS and color not in CSS4_COLORS:
            raise ValueError(f"Invalid color: {color}")

    # Load preset (dataset and its task class)
    preset_class = None
    for v in importlib.import_module(f"data.{args.preset}").__dict__.values():
        if inspect.isclass(v) and issubclass(v, Preset) and v is not Preset:
            preset_class = v
            break
    if preset_class is None:
        raise ValueError(f"Invalid preset: {args.preset}")
    preset: Preset = preset_class()
    print("Preset:", preset.name)

    # Analyze given parameters and create parameter groups
    assert (
        len(
            set(
                [
                    1,
                    len(args.alpha),
                    len(args.r),
                    len(args.gamma),
                    len(args.nu),
                    len(args.lambda_max),
                ]
            )
        )
        < 3  # ♡
    ), "All parameters must have the same length or 1."

    def get_arg_list(name: str, group_idx: int) -> List[float]:
        if len(args.__dict__[name]) == 1:
            return flatten_list(args.__dict__[name][0])
        return flatten_list(args.__dict__[name][group_idx])

    param_dicts: List[Dict[str, List[float]]] = []
    for group_idx in range(len(args.alpha)):
        param_dict: Dict[str, List[float]] = {
            "alpha": get_arg_list("alpha", group_idx),
            "r": get_arg_list("r", group_idx),
            "gamma": get_arg_list("gamma", group_idx),
            "nu": get_arg_list("nu", group_idx),
            "lambda_max": get_arg_list("lambda_max", group_idx),
        }
        print(f"Group [ {group_idx + 1} / {len(args.alpha)} ] parameters:")
        print(param_dict)
        param_dicts.append(param_dict)

    # Run experiments
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    for group_idx in range(len(args.alpha)):
        print(
            f"Running group [ {group_idx + 1} / {len(args.alpha)} ]",
            flush=True,
        )

        # Save parameters
        param_dict = param_dicts[group_idx]
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            os.path.join(
                args.output_dir,
                f"{run_name}_{preset.name}_{group_idx}_params.yaml",
            ),
            "w",
        ) as fs:
            yaml.dump(param_dict, fs)

        # Run experiments and save metrics (I_alpha and err)
        print("  - Running experiments...", flush=True, end=" ")
        start_time = time.time()
        exp = Experiment(
            preset.applicable_task,
            args.thr_finding_granularity,
        )
        exp.task.train(*preset.get_train_data())
        exp.task.test(*preset.get_test_data())
        exp_results, exp_metrics = exp.run(
            param_dict,
            args.plot_convergence,
            delete_results=(not args.plot_convergence),
            return_metrics=True,
            metrics_repeat_times=args.test_repeat_times,
        )
        print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)

        print("  - Saving metrics...", flush=True, end=" ")
        start_time = time.time()
        metrics_pkl_path = os.path.join(
            args.output_dir,
            f"{run_name}_{preset.name}_{group_idx}_metrics.pkl",
        )
        with open(metrics_pkl_path, "wb") as f:
            pickle.dump(exp_metrics, f)
        print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)
        print(f"    saved: {metrics_pkl_path}", flush=True)

        # Plot and save metrics (I_alpha and err, by gamma, if applicable)
        assert exp_metrics is not None, "exp_metrics should not be None"
        gamma_values = set()
        for exp_key in exp_metrics.keys():
            gamma_values.add(frozenkey_to_paramdict(exp_key)["gamma"])
        if len(gamma_values) < 2:
            print("  - Skipping plotting by gamma due to insufficient params.")
        else:
            print("  - Plotting I_alpha ...", flush=True, end=" ")
            start_time = time.time()
            I_alpha_pdf_path = os.path.join(
                args.output_dir,
                f"{run_name}_{preset.name}_{group_idx}_I_alpha.pdf",
            )
            fig = plot_metrics(
                exp_metrics,
                "I_alpha",
                save_path=I_alpha_pdf_path,
                color=args.colors,
                figsize=args.figsize,
            )
            fig.clear()
            plt.close(fig)
            print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)
            print(f"    saved: {I_alpha_pdf_path}", flush=True)

            print("  - Plotting err ...", flush=True, end=" ")
            start_time = time.time()
            err_pdf_path = os.path.join(
                args.output_dir,
                f"{run_name}_{preset.name}_{group_idx}_err.pdf",
            )
            fig = plot_metrics(
                exp_metrics,
                "err",
                save_path=err_pdf_path,
                color=args.colors,
                figsize=args.figsize,
            )
            fig.clear()
            plt.close(fig)
            print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)
            print(f"    saved: {err_pdf_path}", flush=True)

        # Plot and save convergence traces (if applicable)
        if args.plot_convergence:
            assert exp_results is not None, "exp_results should not be None"
            print("  - Plotting convergence traces ...", flush=True, end=" ")
            start_time = time.time()
            I_alpha_trace_pdf_path = os.path.join(
                args.output_dir,
                f"{run_name}_{preset.name}_{group_idx}_I_alpha_trace.pdf",
            )
            fig = plot_convergence(
                exp_results,
                "I_alpha",
                save_path=I_alpha_trace_pdf_path,
                color=args.colors,
                figsize=args.figsize,
            )
            fig.clear()
            plt.close(fig)
            print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)
            print(f"    saved: {I_alpha_trace_pdf_path}", flush=True)

            print("  - Plotting err traces ...", flush=True, end=" ")
            start_time = time.time()
            err_trace_pdf_path = os.path.join(
                args.output_dir,
                f"{run_name}_{preset.name}_{group_idx}_err_trace.pdf",
            )
            fig = plot_convergence(
                exp_results,
                "err",
                save_path=err_trace_pdf_path,
                color=args.colors,
                figsize=args.figsize,
            )
            fig.clear()
            plt.close(fig)
            print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)
            print(f"    saved: {err_trace_pdf_path}", flush=True)

        # Clean up
        print("  - Cleaning up...", flush=True, end=" ")
        start_time = time.time()
        del exp_metrics
        del exp_results
        del exp
        gc.collect()
        print(f"done in ({time.time() - start_time:.2f} sec)", flush=True)

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        action="append",
        type=versatile_float,
        nargs="+",
    )
    parser.add_argument(
        "--r",
        action="append",
        type=versatile_float,
        nargs="+",
    )
    parser.add_argument(
        "--gamma",
        action="append",
        type=versatile_float,
        nargs="+",
    )
    parser.add_argument(
        "--nu",
        action="append",
        type=versatile_float,
        nargs="+",
    )
    parser.add_argument(
        "--lambda_max",
        action="append",
        type=versatile_float,
        nargs="+",
    )

    parser.add_argument("--thr_finding_granularity", type=int, default=200)
    parser.add_argument("--test_repeat_times", type=int, default=10000)

    parser.add_argument("--plot_convergence", action="store_true")

    parser.add_argument(
        "--colors",
        type=str,
        action="append",
        default=plotting_default_colors,
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=plotting_default_figsize,
    )

    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()
    main(args)
