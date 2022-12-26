from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .algorithm_gefair import GEFairResult
from .experiment import Metrics
from .utils import FrozenKey, average_by_time


def plot_metrics(
    exp_metrics: Dict[FrozenKey, Metrics],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = (8, 6),
    color: Union[List[str], str] = "black",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot metrics"""

    assert metric_name in ["I_alpha", "err"]

    experiments_to_draw: Dict[FrozenKey, Metrics] = {}
    r_values = set()
    for exp_key, exp_metric in exp_metrics.items():
        param_dict = {k: v for k, v in list(exp_key)}
        if not params_filter(param_dict):
            continue
        experiments_to_draw[exp_key] = exp_metric
        r_values.add(param_dict["r"])

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if type(color) == str:
        colors: List[str] = [color] * len(r_values)  # type: ignore
    else:
        assert len(color) == len(
            r_values
        ), "Number of colors must match number of r values"
        colors = color  # type: ignore

    r_values_list = sorted(list(r_values))
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

    if len(r_values) > 1:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig


def plot_convergence(
    exp_results: Dict[FrozenKey, GEFairResult],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = (8, 6),
    color: Union[List[str], str] = "black",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot trace of metrics during training, using time-axis averaging"""

    assert metric_name in ["I_alpha", "err", "threshold", "hypothesis"]

    experiments_to_draw: Dict[FrozenKey, GEFairResult] = {}
    for exp_key, exp_result in exp_results.items():
        param_dict = {k: v for k, v in list(exp_key)}
        if not params_filter(param_dict):
            continue
        experiments_to_draw[exp_key] = exp_result

    fig, ax = plt.subplots(figsize=figsize)

    if len(experiments_to_draw) == 0:
        print("No metrics to draw.")
        return fig

    if type(color) == str:
        colors: List[str] = [color] * len(experiments_to_draw.keys())  # type: ignore
    else:
        assert len(color) == len(
            experiments_to_draw.keys()
        ), "Number of colors must match number of experiments"
        colors = color  # type: ignore

    for expi, (exp_key, exp_result) in enumerate(experiments_to_draw.items()):
        param_dict = {k: v for k, v in list(exp_key)}
        key_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
        if metric_name == "I_alpha":
            ax.plot(
                average_by_time(exp_result.I_alpha_history),
                label=key_str,
                color=colors[expi],
            )
        elif metric_name == "err":
            ax.plot(
                average_by_time(exp_result.err_history),
                label=key_str,
                color=colors[expi],
            )
        elif metric_name == "threshold" or metric_name == "hypothesis":
            ax.plot(
                average_by_time(exp_result.hypothesis_history),
                label=key_str,
                color=colors[expi],
            )
        else:
            continue

    if len(experiments_to_draw.keys()) > 1:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig
