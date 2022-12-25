from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .experiment import Metrics
from .utils import FrozenKey


def plot_line(
    exp_metrics: Dict[FrozenKey, Metrics],
    metric_name: str,
    params_filter: Callable[[Dict[str, float]], bool] = lambda _: True,
    figsize: Tuple[float, float] = (8, 6),
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

    for r in r_values:
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
            ax.errorbar(plot_x, plot_y, yerr=plot_err, label=f"r={r}")
        else:
            ax.plot(plot_x, plot_y, "o-", label=f"r={r}")

    if len(r_values) > 1:
        ax.legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig


def plot_band():
    raise NotImplementedError
