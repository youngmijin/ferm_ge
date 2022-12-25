import argparse
import importlib
import inspect
import os
import pickle
import time
from typing import Dict, List

import numpy as np

from data import Preset
from ferm_ge import Experiment, plot_line


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
    preset_class = None
    for v in importlib.import_module(f"data.{args.preset}").__dict__.values():
        if inspect.isclass(v) and issubclass(v, Preset) and v is not Preset:
            preset_class = v
            break
    if preset_class is None:
        raise ValueError(f"Invalid preset: {args.preset}")

    preset: Preset = preset_class()

    print("Preset:", preset.name)

    assert (
        len(args.alpha)
        == len(args.r)
        == len(args.gamma)
        == len(args.nu)
        == len(args.lambda_max)
    ), "All parameters must have the same length."

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    for group_idx in range(len(args.alpha)):
        print(f"Group {group_idx + 1}/{len(args.alpha)} ...", flush=True)

        exp = Experiment(
            preset.applicable_task,
            args.thr_finding_granularity,
        )

        param_dict: Dict[str, List[float]] = {
            "alpha": flatten_list(args.alpha[group_idx]),
            "r": flatten_list(args.r[group_idx]),
            "gamma": flatten_list(args.gamma[group_idx]),
            "nu": flatten_list(args.nu[group_idx]),
            "lambda_max": flatten_list(args.lambda_max[group_idx]),
        }

        with open(
            os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_params.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(param_dict, f)

        print("Parameters:")
        print(param_dict, flush=True)

        exp.task.train(*preset.get_train_data())
        exp.task.test(*preset.get_test_data())

        exp_results = exp.solve(param_dict)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_results.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(exp_results, f)

        exp_metrics = exp.get_metrics_with_prob(exp_results)
        with open(
            os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_metrics.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(exp_metrics, f)

        print("Drawing plots ...", flush=True, end=" ")

        plot_line(
            exp_metrics,
            "I_alpha",
            save_path=os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_Ialpha.pdf",
            ),
        )
        plot_line(
            exp_metrics,
            "err",
            save_path=os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_err.pdf",
            ),
        )

        print(f"group {group_idx + 1} done.", flush=True)

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

    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()
    main(args)
