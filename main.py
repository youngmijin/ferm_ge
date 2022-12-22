import argparse
import importlib
import inspect
import os
import pickle
import time
from typing import Dict, List

from data import Preset
from ferm_ge import Experiment


def main(args):
    preset_class = None
    for v in importlib.import_module(f"data.{args.preset}").__dict__.values():
        if inspect.isclass(v) and issubclass(v, Preset) and v is not Preset:
            preset_class = v
            break
    if preset_class is None:
        raise ValueError(f"Invalid preset: {args.preset}")

    preset: Preset = preset_class()
    if not preset.check_files():
        preset.download()
    preset.load_files()

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
        print(f"Group {group_idx + 1}/{len(args.alpha)} ...")

        exp = Experiment(preset.applicable_task)

        param_dict: Dict[str, List[float]] = {
            "alpha": args.alpha[group_idx],
            "r": args.r[group_idx],
            "gamma": args.gamma[group_idx],
            "nu": args.nu[group_idx],
            "lambda_max": args.lambda_max[group_idx],
        }

        print("Parameters:")
        print(param_dict)

        exp.task.train(*preset.get_train_data())
        exp.task.test(*preset.get_test_data())

        exp_results = exp.solve(param_dict)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(
            os.path.join(
                args.output_dir, f"{current_time}_{preset.name}_{group_idx}.pkl"
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

        exp.plot_metrics(
            exp_metrics,
            "I_alpha",
            save_path=os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_Ialpha.pdf",
            ),
        )
        exp.plot_metrics(
            exp_metrics,
            "err",
            save_path=os.path.join(
                args.output_dir,
                f"{current_time}_{preset.name}_{group_idx}_err.pdf",
            ),
        )

    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", action="append", type=float, nargs="+")
    parser.add_argument("--r", action="append", type=float, nargs="+")
    parser.add_argument("--gamma", action="append", type=float, nargs="+")
    parser.add_argument("--nu", action="append", type=float, nargs="+")
    parser.add_argument("--lambda_max", action="append", type=float, nargs="+")

    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()
    main(args)
