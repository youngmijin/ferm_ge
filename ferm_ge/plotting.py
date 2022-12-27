import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .algorithm_gefair import GEFairResultSM
from .experiment import Metrics
from .utils import FrozenKey, average_by_time

matplotlib.rcParams["font.family"] = "serif"

default_colors = [
    "red",
    "green",
    "black",
    "orange",
    "navy",
    "violet",
]

default_figsize = (4, 2.3)


def plot_metrics(
    exp_metrics: Dict[FrozenKey, Metrics],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = default_figsize,
    color: Union[List[str], str] = default_colors,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    assert metric_name in ["I_alpha", "err"]

    experiments_to_draw: Dict[FrozenKey, Metrics] = {}
    r_values = set()
    alpha_values = set()
    for exp_key, exp_metric in exp_metrics.items():
        param_dict = {k: v for k, v in list(exp_key)}
        if not params_filter(param_dict):
            continue
        experiments_to_draw[exp_key] = exp_metric
        r_values.add(param_dict["r"])
        alpha_values.add(param_dict["alpha"])

    if len(alpha_values) > 1:
        warnings.warn(
            f"Multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if type(color) == str:
        colors: List[str] = [color] * len(r_values)  # type: ignore
    else:
        assert len(color) >= len(
            r_values
        ), "Number of colors must match or higher than the number of r values"
        colors = color  # type: ignore

    r_values_list = sorted(list(r_values))
    xmin = float("inf")
    xmax = float("-inf")
    for ri, r in enumerate(r_values_list):
        plot_x = []
        plot_y = []
        plot_err = []
        for exp_key, exp_metric in experiments_to_draw.items():
            param_dict = {k: v for k, v in list(exp_key)}
            if param_dict["r"] != r:
                continue
            plot_x.append(param_dict["gamma"])
            plot_y.append(getattr(exp_metric, metric_name))
            if getattr(exp_metric, f"{metric_name}_std") is not None:
                plot_err.append(getattr(exp_metric, f"{metric_name}_std"))
        if len(plot_err) > 0:
            plot_err = [e * 1.96 for e in plot_err]  # 95% confidence interval
            ax.plot(plot_x, plot_y, label=f"r={r}", color=colors[ri])
            ax.fill_between(
                plot_x,
                [y - e for y, e in zip(plot_y, plot_err)],  # type: ignore
                [y + e for y, e in zip(plot_y, plot_err)],  # type: ignore
                color=colors[ri],
                alpha=0.2,
            )
        else:
            ax.plot(plot_x, plot_y, "o-", label=f"r={r}", color=colors[ri])
        xmin = min(xmin, min(plot_x))
        xmax = max(xmax, max(plot_x))

    if len(r_values) > 1:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    if xmin != xmax:
        ax.set_xlim(xmin, xmax)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig


def plot_convergence(
    exp_results: Dict[FrozenKey, GEFairResultSM],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = default_figsize,
    color: Union[List[str], str] = default_colors,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot trace of metrics during training, using time-axis averaging"""

    assert metric_name in ["I_alpha", "err", "threshold", "hypothesis"]

    experiments_to_draw: Dict[FrozenKey, GEFairResultSM] = {}
    r_values = set()
    alpha_values = set()
    for exp_key, exp_result in exp_results.items():
        param_dict = {k: v for k, v in list(exp_key)}
        if not params_filter(param_dict):
            continue
        experiments_to_draw[exp_key] = exp_result
        r_values.add(param_dict["r"])
        alpha_values.add(param_dict["alpha"])

    if len(alpha_values) > 1:
        warnings.warn(
            f"Multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if type(color) == str:
        colors: List[str] = [color] * len(r_values)  # type: ignore
    else:
        assert len(color) >= len(
            r_values
        ), "Number of colors must match or higher than the number of experiments"
        colors = color  # type: ignore

    r_values_list = sorted(list(r_values))
    for ri, r in enumerate(r_values_list):
        for exp_key, exp_result in experiments_to_draw.items():
            param_dict = {k: v for k, v in list(exp_key)}
            if param_dict["r"] != r:
                continue
            things_to_plot: Optional[np.ndarray] = None
            if metric_name == "I_alpha":
                assert (
                    exp_result.I_alpha_history is not None
                ), "I_alpha_history is None"
                things_to_plot = average_by_time(exp_result.I_alpha_history)
            elif metric_name == "err":
                assert exp_result.err_history is not None, "err_history is None"
                things_to_plot = average_by_time(exp_result.err_history)
            elif metric_name == "threshold" or metric_name == "hypothesis":
                assert (
                    exp_result.hypothesis_history is not None
                ), "hypothesis_history is None"
                things_to_plot = average_by_time(exp_result.hypothesis_history)
            assert things_to_plot is not None, "things_to_plot is None"
            ax.plot(things_to_plot, label=f"r={r}", color=colors[ri])

    if len(r_values) > 1:
        ax.legend(loc="upper right")

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig
