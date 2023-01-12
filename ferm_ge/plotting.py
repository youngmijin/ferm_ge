import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib._color_data import BASE_COLORS, CSS4_COLORS
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .exp import ExpTestResult, ExpTrainResult, ParamSet

plt.rcParams["ps.useafm"] = True
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["text.usetex"] = True

DEFAULT_FIGSIZE = (3, 1.5)
DEFAULT_COLORS = ["black"]
DEFAULT_STYLES = ["solid"]


def __check_arguments(
    ps_list: list[ParamSet],
    metric_names: str | list[str],
    valid_metric_names: list[str],
    colors: str | list[str],
    styles: str | list[str],
) -> tuple[list[str], list[str], list[str], list[float]]:
    alpha_values = sorted(set([ps.alpha for ps in ps_list]))
    c_values = sorted(set([ps.c for ps in ps_list]))
    a_values = sorted(set([ps.a for ps in ps_list]))

    if isinstance(metric_names, str):
        metric_names = [metric_names]
    for metric_name in metric_names:
        assert (
            metric_name in valid_metric_names
        ), f"unknown metric name: {metric_name}"

    if isinstance(colors, str):
        colors = [colors] * len(c_values)
    elif len(colors) == 1:
        colors = colors * len(c_values)
    else:
        assert len(colors) >= len(c_values), "not enough colors"
    for color in colors:
        assert (
            color in BASE_COLORS or color in CSS4_COLORS
        ), f"unknown color: {color}"

    if isinstance(styles, str):
        styles = [styles] * len(metric_names)
    elif len(styles) == 1:
        styles = styles * len(metric_names)
    else:
        assert len(styles) >= len(metric_names), "not enough styles"
    for style in styles:
        assert style in [
            "solid",
            "dashed",
            "dashdot",
            "dotted",
        ], f"unknown style: {style}"

    if len(alpha_values) > 1:
        warnings.warn(
            f"multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )
    if len(a_values) > 1:
        warnings.warn(f"multiple values of a detected: {a_values}", UserWarning)

    return metric_names, colors, styles, c_values


def __resample(
    array: NDArray[np.float_],
    sampling_threshold: int | None = 2000000,
    sampling_exclude_initial: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    assert array.ndim == 1, "array must be 1D array"
    if sampling_threshold is None or len(array) <= sampling_threshold:
        return np.arange(len(array)), array
    else:
        former_x = np.arange(sampling_exclude_initial)
        former_y = array[:sampling_exclude_initial]
        latter_x = np.arange(
            sampling_exclude_initial,
            len(array),
            len(array) // sampling_threshold,
        )
        latter_y = array[
            sampling_exclude_initial :: len(array) // sampling_threshold
        ]
        return (
            np.concatenate([former_x, latter_x]),
            np.concatenate([former_y, latter_y]),
        )


def plot_test_results_by_gamma(
    results: dict[ParamSet, ExpTestResult],
    metric_names: str | list[str],
    baselines: dict[ParamSet, dict[str, float]] | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    colors: str | list[str] = DEFAULT_COLORS,
    styles: str | list[str] = DEFAULT_STYLES,
) -> Figure:
    ps_list = list(results.keys())
    metric_names, colors, styles, c_values = __check_arguments(
        ps_list,
        metric_names,
        ["ge", "err", "mseo", "aseo"],
        colors,
        styles,
    )

    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax = float("inf"), float("-inf")
    ymin, ymax = float("inf"), float("-inf")
    baselines_by_ci: dict[int, list[float]] = {}
    for ci, c in enumerate(c_values):
        for mi, metric_name in enumerate(metric_names):
            data_x: list[float] = []
            data_y: list[float] = []
            data_e: list[float] = []
            for ps, result in results.items():
                if ps.c != c:
                    continue
                if metric_name == "ge":
                    data_y.append(result.ge)
                    data_e.append(result.ge_std)
                elif metric_name == "err":
                    data_y.append(result.err)
                    data_e.append(result.err_std)
                elif metric_name == "mseo":
                    assert (
                        result.mseo is not None and result.mseo_std is not None
                    ), f"mseo is not available for {ps}"
                    data_y.append(result.mseo)
                    data_e.append(result.mseo_std)
                elif metric_name == "aseo":
                    assert (
                        result.aseo is not None and result.aseo_std is not None
                    ), f"aseo is not available for {ps}"
                    data_y.append(result.aseo)
                    data_e.append(result.aseo_std)
                else:
                    continue
                data_x.append(ps.gamma)

                if baselines is not None:
                    if ci not in baselines_by_ci:
                        baselines_by_ci[ci] = []
                    baselines_by_ci[ci].append(baselines[ps][metric_name])

            data_e_upper = np.array(data_y) + np.array(data_e) * 1.96
            data_e_lower = np.array(data_y) - np.array(data_e) * 1.96

            xmin = min(xmin, min(data_x))
            xmax = max(xmax, max(data_x))
            ymin = min(ymin, min(data_e_lower))
            ymax = max(ymax, max(data_e_upper))

            ax.plot(
                data_x,
                data_y,
                label=(
                    f"$c={c}$"
                    if len(metric_names) == 1
                    else f"{metric_name} $c={c}$"
                ),
                color=colors[ci],
                linestyle=styles[mi],
            )
            ax.fill_between(
                data_x,
                data_e_lower,  # type: ignore
                data_e_upper,  # type: ignore
                color=colors[ci],
                alpha=0.2,
            )

    for ci, baseline_values in baselines_by_ci.items():
        for baseline_value in set(baseline_values):
            ax.axhline(
                baseline_value,
                color=colors[ci],
                linestyle="--",
                alpha=0.4,
            )

    if (len(c_values) * len(metric_names)) > 1:
        if "ge" in metric_names:
            ax.legend(loc="lower right")
        else:
            ax.legend(loc="upper right")

    if xmin != xmax:
        ax.set_xlim(xmin, xmax)

    ylim_min = ymin - 0.15 * (ymax - ymin)
    ylim_max = ymax + 0.15 * (ymax - ymin)
    ax.set_ylim(ylim_min, ylim_max)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=600)

    return fig


def plot_training_traces(
    results: dict[ParamSet, ExpTrainResult],
    metric_names: str | list[str],
    magnify: tuple[float, float] | None = None,
    fname: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    colors: str | list[str] = DEFAULT_COLORS,
    styles: str | list[str] = DEFAULT_STYLES,
) -> Figure:
    ps_list = list(results.keys())
    metric_names, colors, styles, c_values = __check_arguments(
        ps_list,
        metric_names,
        ["ge_bar", "err_bar", "mseo", "aseo"],
        colors,
        styles,
    )

    fig, ax = plt.subplots(figsize=figsize)
    for ci, c in enumerate(c_values):
        for mi, metric_name in enumerate(metric_names):
            for ps, result in results.items():
                if ps.c != c:
                    continue
                data: NDArray[np.float_] | None = None
                if metric_name == "ge_bar":
                    assert (
                        result.ge_bar_trace is not None
                    ), f"ge_bar_trace is not available for {ps}"
                    data = result.ge_bar_trace
                elif metric_name == "err_bar":
                    assert (
                        result.err_bar_trace is not None
                    ), f"err_bar_trace is not available for {ps}"
                    data = result.err_bar_trace
                elif metric_name == "mseo":
                    assert (
                        result.mseo_trace is not None
                    ), f"mseo_trace is not available for {ps}"
                    data = result.mseo_trace
                elif metric_name == "aseo":
                    assert (
                        result.aseo_trace is not None
                    ), f"aseo_trace is not available for {ps}"
                    data = result.aseo_trace
                if data is None:
                    continue
                xs, ys = __resample(data)
                ax.plot(
                    xs,
                    ys,
                    label=(
                        f"$c={c}$"
                        if len(metric_names) == 1
                        else f"{metric_name} $c={c}$"
                    ),
                    color=colors[ci],
                    linestyle=styles[mi],
                )

    if (len(c_values) * len(metric_names)) > 1:
        if "ge_bar" in metric_names:
            ax.legend(loc="lower right")
        else:
            ax.legend(loc="upper right")

    if magnify is not None:
        ax.set_xlim(*magnify)

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=600)

    return fig
