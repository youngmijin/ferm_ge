import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .algorithm_gefair import GEFairResultSM
from .experiment import BaselineValues
from .metrics import Metrics
from .utils import FrozenKey, apply_sampling, frozenkey_to_paramdict

matplotlib.rcParams["font.family"] = "serif"

default_color = "black"
default_figsize = (4, 2.3)


def plot_metrics(
    exp_metrics: Dict[FrozenKey, Metrics],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = default_figsize,
    color: Union[List[str], str] = default_color,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    baseline: Optional[Dict[FrozenKey, BaselineValues]] = None,
) -> Figure:
    assert metric_name in ["I_alpha", "err"]

    experiments_to_draw: List[FrozenKey] = []
    alpha_values = set()
    c_values = set()
    a_values = set()
    for exp_key, exp_metric in exp_metrics.items():
        param_dict = frozenkey_to_paramdict(exp_key)
        if not params_filter(param_dict):
            continue
        experiments_to_draw.append(exp_key)
        alpha_values.add(param_dict["alpha"])
        c_values.add(param_dict["c"])
        a_values.add(param_dict["a"])

    if len(alpha_values) > 1:
        warnings.warn(
            f"Multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )
    if len(a_values) > 1:
        warnings.warn(f"Multiple values of a detected: {a_values}", UserWarning)

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if type(color) == str:
        colors: List[str] = [color] * len(r_values)  # type: ignore
    else:
        assert len(color) >= len(
            c_values
        ), "Number of colors must match or higher than the number of r values"
        colors = color  # type: ignore

    c_values_list = sorted(list(c_values))
    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")
    for ci, c in enumerate(c_values_list):
        plot_x = []
        plot_y = []
        plot_err = []
        for exp_key, exp_metric in exp_metrics.items():
            if exp_key not in experiments_to_draw:
                continue
            param_dict = frozenkey_to_paramdict(exp_key)
            if param_dict["c"] != c:
                continue
            plot_x.append(param_dict["gamma"])
            plot_y.append(getattr(exp_metric, metric_name))
            if getattr(exp_metric, f"{metric_name}_std") is not None:
                plot_err.append(getattr(exp_metric, f"{metric_name}_std"))
        if len(plot_err) > 0:
            plot_err = [e * 1.96 for e in plot_err]  # 95% confidence interval
            ax.plot(plot_x, plot_y, label=f"c={c}", color=colors[ci])
            plot_err_l = [y - e for y, e in zip(plot_y, plot_err)]
            plot_err_u = [y + e for y, e in zip(plot_y, plot_err)]
            ax.fill_between(
                plot_x,
                plot_err_l,  # type: ignore
                plot_err_u,  # type: ignore
                color=colors[ci],
                alpha=0.2,
            )
            ymin = min(ymin, min(plot_err_l))
            ymax = max(ymax, max(plot_err_u))
        else:
            ax.plot(plot_x, plot_y, "o-", label=f"c={c}", color=colors[ci])
            ymin = min(ymin, min(plot_y))
            ymax = max(ymax, max(plot_y))
        xmin = min(xmin, min(plot_x))
        xmax = max(xmax, max(plot_x))

    if baseline is not None:
        baseline_dict = {}
        for exp_key, baseline_value in baseline.items():
            if exp_key not in experiments_to_draw:
                continue
            param_dict = frozenkey_to_paramdict(exp_key)
            baseline_dict[param_dict["c"]] = getattr(
                baseline_value, metric_name
            )

        for ci, c in enumerate(c_values_list):
            assert c in baseline_dict, f"Missing baseline for c={c}"
            baseline_value = baseline_dict[c]
            ax.axhline(
                baseline_value, linestyle="--", color=colors[ci], alpha=0.4
            )

    if len(c_values) > 1:
        if metric_name == "I_alpha":
            ax.legend(loc="lower right")
        elif metric_name == "err":
            ax.legend(loc="upper right")

    if title is not None:
        ax.set_title(title)

    if xmin != xmax:
        ax.set_xlim(xmin, xmax)

    ylim_min = ymin - 0.15 * (ymax - ymin)
    ylim_max = ymax + 0.15 * (ymax - ymin)
    ax.set_ylim(ylim_min, ylim_max)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig


def plot_convergence(
    exp_results: Dict[FrozenKey, GEFairResultSM],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = default_figsize,
    color: Union[List[str], str] = default_color,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    sampling_threshold: Optional[int] = 2000000,
    sampling_exclude_initial: int = 10000,
    magnify: Optional[Tuple[float, float]] = None,
    highlight_range: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Plot trace of metrics during training, using time-axis averaging"""

    assert metric_name in ["I_alpha", "err", "threshold", "hypothesis"]

    experiments_to_draw: List[FrozenKey] = []
    alpha_values = set()
    c_values = set()
    a_values = set()
    for exp_key in exp_results.keys():
        param_dict = {k: v for k, v in list(exp_key)}
        if not params_filter(param_dict):
            continue
        experiments_to_draw.append(exp_key)
        alpha_values.add(param_dict["alpha"])
        c_values.add(param_dict["c"])
        a_values.add(param_dict["a"])

    if len(alpha_values) > 1:
        warnings.warn(
            f"Multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )
    if len(a_values) > 1:
        warnings.warn(f"Multiple values of a detected: {a_values}", UserWarning)

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if highlight_range is not None:
        ax.axvspan(*highlight_range, color="bisque")

    if type(color) == str:
        colors: List[str] = [color] * len(r_values)  # type: ignore
    else:
        assert len(color) >= len(
            c_values
        ), "Number of colors must match or higher than the number of experiments"
        colors = color  # type: ignore

    c_values_list = sorted(list(c_values))
    for ci, c in enumerate(c_values_list):
        for exp_key, exp_result in exp_results.items():
            if exp_key not in experiments_to_draw:
                continue
            param_dict = {k: v for k, v in list(exp_key)}
            if param_dict["c"] != c:
                continue
            things_to_plot: Optional[np.ndarray] = None
            if metric_name == "I_alpha":
                assert (
                    exp_result.I_alpha_bar is not None
                ), "I_alpha_history is None"
                things_to_plot = exp_result.I_alpha_bar
            elif metric_name == "err":
                assert exp_result.err_bar is not None, "err_history is None"
                things_to_plot = exp_result.err_bar
            elif metric_name == "threshold" or metric_name == "hypothesis":
                assert (
                    exp_result.D_bar is not None
                ), "hypothesis_history is None"
                things_to_plot = exp_result.D_bar
            assert things_to_plot is not None, "things_to_plot is None"
            plot_x, plot_y = apply_sampling(
                things_to_plot,
                sampling_threshold,
                sampling_exclude_initial,
            )
            ax.plot(
                plot_x,
                plot_y,
                label=f"c={c}",
                color=colors[ci],
            )

    if len(c_values) > 1:
        ax.legend(loc="upper right")

    if magnify is not None:
        ax.set_xlim(*magnify)

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig
