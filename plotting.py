import statistics
import warnings
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ferm_ge import ExpTrainResult, ExpValidResult, ParamSet

DEFAULT_FIGSIZE = (3, 1.5)
DEFAULT_LINECOLOR = "black"
DEFAULT_LINESTYLE = "solid"

PLOTABLE_METRICS = [
    "ge",
    "err",
    "aseo",
    "aseo_fp",
    "aseo_fn",
    "v",
    "group_rfp",
    "group_rfn",
]

METRIC2TEX_DICT = {
    "ge": "$I_\\alpha$",
    "ge_bar": "$\\bar{{I}}_\\alpha$",
    "err": "error",
    "err_bar": "$\\bar{{\\mathrm{{error}}}}$",
    "aseo": "ASEO",
    "aseo_fp": "FP",
    "aseo_fn": "FN",
    "v": "$V$",
    "group_rfp": "{group_name} $r^{{FP}}$",
    "group_rfn": "{group_name} $r^{{FN}}$",
}


def __legend(m: str, d: dict[str, str] | None = None) -> str:
    if d is None:
        return METRIC2TEX_DICT[m]
    return METRIC2TEX_DICT[m].format(**d)


def __check_arguments(
    ps_list: list[ParamSet],
    metric_names: list[str],
    metric_names_mappings: dict[str, str],
    use_tex: bool,
) -> tuple[list[str], list[float]]:
    alpha_values = sorted(set([ps.alpha for ps in ps_list]))
    c_values = sorted(set([ps.c for ps in ps_list]))
    a_values = sorted(set([ps.a for ps in ps_list]))

    for metric_name in metric_names:
        assert (
            metric_name in PLOTABLE_METRICS
        ), f"unknown metric name: {metric_name}"

    if len(alpha_values) > 1:
        warnings.warn(
            f"multiple values of alpha detected: {alpha_values}",
            UserWarning,
        )
    if len(a_values) > 1:
        warnings.warn(f"multiple values of a detected: {a_values}", UserWarning)

    metric_names = [
        (metric_names_mappings[mn] if mn in metric_names_mappings else mn)
        for mn in metric_names
    ]

    if use_tex:
        plt.rcParams["ps.useafm"] = True
        plt.rcParams["pdf.use14corefonts"] = True
        plt.rcParams["text.usetex"] = True

    return metric_names, c_values


def __resample(
    array: NDArray[np.float_],
    sampling_threshold: int | None = 2000000,
    exclude_first: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    assert array.ndim == 1, "array must be 1D array"
    if sampling_threshold is None or len(array) <= sampling_threshold:
        return np.arange(len(array)), array
    else:
        assert sampling_threshold > 0, "sampling_threshold must be positive"
        former_x = np.arange(exclude_first)
        former_y = array[:exclude_first]
        latter_x = np.arange(
            exclude_first,
            len(array),
            len(array) // sampling_threshold,
        )
        latter_y = array[exclude_first :: len(array) // sampling_threshold]
        return (
            np.concatenate([former_x, latter_x]),
            np.concatenate([former_y, latter_y]),
        )


def plot_valid_results_by_gamma_c(
    results: dict[ParamSet, ExpValidResult],
    metrics: list[str],
    metrics_right: list[str] = [],
    group_names: list[str] = ["unknown"],
    baselines: dict[ParamSet, dict[str, float]] | None = None,
    confidence_band: float = 0.0,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    coloring_rules: defaultdict[str, str | None] = defaultdict(lambda: None),
    styling_rules: defaultdict[str, str | None] = defaultdict(lambda: None),
    use_tex: bool = False,
) -> Figure:
    ps_list = list(results.keys())
    metrics, c_values = __check_arguments(ps_list, metrics, {}, use_tex)

    assert (
        len(metrics) > 0 and len(group_names) > 0
    ), "metrics and group_names must not be empty"
    assert (
        confidence_band >= 0 and confidence_band < 1
    ), "confidence_band must be in [0, 1)"
    z_value = statistics.NormalDist().inv_cdf((1 + confidence_band) / 2)

    fig, axl = plt.subplots(figsize=figsize)
    axr = axl.twinx() if len(metrics_right) > 0 else axl
    xmin, xmax = float("inf"), float("-inf")
    lymin, lymax = float("inf"), float("-inf")
    rymin, rymax = float("inf"), float("-inf")
    baselines_by_ci_for_axl: dict[int, list[float]] = {
        ci: [] for ci in range(len(c_values))
    }
    baselines_by_ci_for_axr: dict[int, list[float]] = {
        ci: [] for ci in range(len(c_values))
    }
    for ci, c in enumerate(c_values):
        for metric_name, group_name in product(metrics, group_names):
            data_x: list[float] = []
            data_y: list[float] = []
            data_e: list[float] = []
            for ps, result in results.items():
                if ps.c != c:
                    continue
                if metric_name == "ge":
                    mean, std = result.ge
                elif metric_name == "err":
                    mean, std = result.err
                elif metric_name == "aseo":
                    assert result.aseo is not None
                    mean, std = result.aseo
                elif metric_name == "aseo_fp":
                    assert result.aseo_fp is not None
                    mean, std = result.aseo_fp
                elif metric_name == "aseo_fn":
                    assert result.aseo_fn is not None
                    mean, std = result.aseo_fn
                elif metric_name == "v":
                    assert result.v is not None
                    mean, std = result.v
                elif metric_name == "group_rfp":
                    assert result.group_rfp is not None
                    assert group_name in result.group_rfp
                    mean, std = result.group_rfp[group_name]
                elif metric_name == "group_rfn":
                    assert result.group_rfn is not None
                    assert group_name in result.group_rfn
                    mean, std = result.group_rfn[group_name]
                else:
                    raise ValueError(f"unknown metric: {metric_name}")
                data_x.append(ps.gamma)
                data_y.append(mean)
                data_e.append(std)

                if (
                    (baselines is not None)
                    and (ps in baselines)
                    and (metric_name in baselines[ps])
                ):
                    if metric_name in metrics_right:
                        baselines_by_ci_for_axr[ci].append(
                            baselines[ps][metric_name]
                        )
                    else:
                        baselines_by_ci_for_axl[ci].append(
                            baselines[ps][metric_name]
                        )

            data_e_upper = np.array(data_y) + np.array(data_e) * z_value
            data_e_lower = np.array(data_y) - np.array(data_e) * z_value

            xmin = min(xmin, min(data_x))
            xmax = max(xmax, max(data_x))

            if metric_name in metrics_right:
                rymin = min(rymin, min(data_e_lower))
                rymax = max(rymax, max(data_e_upper))
            else:
                lymin = min(lymin, min(data_e_lower))
                lymax = max(lymax, max(data_e_upper))

            metric_legend = __legend(metric_name, {"group_name": group_name})
            if len(metrics) > 1:
                if len(c_values) == 1:
                    plot_label = metric_legend
                else:
                    plot_label = f"{metric_legend} $c={c}$"
            else:
                plot_label = f"$c={c}$"

            plot_color = (
                coloring_rules[f"c={c}"]
                or coloring_rules[f"m={metric_name}"]
                or coloring_rules[f"g={group_name}"]
                or DEFAULT_LINECOLOR
            )
            plot_linestyle = (
                styling_rules[f"c={c}"]
                or styling_rules[f"m={metric_name}"]
                or coloring_rules[f"g={group_name}"]
                or DEFAULT_LINESTYLE
            )

            (axr if metric_name in metrics_right else axl).plot(
                data_x,
                data_y,
                label=plot_label,
                color=plot_color,
                linestyle=plot_linestyle,
            )
            if confidence_band > 0:
                (axr if metric_name in metrics_right else axl).fill_between(
                    data_x,
                    data_e_lower,  # type: ignore
                    data_e_upper,  # type: ignore
                    color=plot_color,
                    alpha=0.2,
                )

            if metric_name in metrics_right:
                axl.plot(
                    [],
                    [],
                    color=plot_color,
                    linestyle=plot_linestyle,
                    label=plot_label,
                )

    for ci, baseline_values in baselines_by_ci_for_axl.items():
        for baseline_value in set(baseline_values):
            axl.axhline(
                baseline_value,
                color=coloring_rules[f"c={c_values[ci]}"] or DEFAULT_LINECOLOR,
                linestyle="--",
                alpha=0.4,
            )
    for ci, baseline_values in baselines_by_ci_for_axr.items():
        for baseline_value in set(baseline_values):
            axr.axhline(
                baseline_value,
                color=coloring_rules[f"c={c_values[ci]}"] or DEFAULT_LINECOLOR,
                linestyle="--",
                alpha=0.4,
            )

    if (len(c_values) * len(metrics)) > 1:
        if "ge" in metrics:
            axl.legend(loc="lower right")
        else:
            axl.legend(loc="upper right")

    if xmin != xmax:
        axl.set_xlim(xmin, xmax)

    lylim_min = lymin - 0.15 * (lymax - lymin)
    lylim_max = lymax + 0.15 * (lymax - lymin)
    axl.set_ylim(lylim_min, lylim_max)

    if len(metrics_right) > 0:
        rylim_min = rymin - 0.15 * (rymax - rymin)
        rylim_max = rymax + 0.15 * (rymax - rymin)
        axr.set_ylim(rylim_min, rylim_max)

    return fig


def plot_training_traces_by_c(
    results: dict[ParamSet, ExpTrainResult],
    metrics: list[str],
    metrics_right: list[str] = [],
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    coloring_rules: defaultdict[str, str | None] = defaultdict(lambda: None),
    styling_rules: defaultdict[str, str | None] = defaultdict(lambda: None),
    use_tex: bool = False,
) -> Figure:
    ps_list = list(results.keys())
    metrics, c_values = __check_arguments(
        ps_list,
        metrics,
        {"ge": "ge_bar", "err": "err_bar"},
        use_tex,
    )

    fig, axl = plt.subplots(figsize=figsize)
    axr = axl.twinx() if len(metrics_right) > 0 else axl
    for c in c_values:
        for metric_name in metrics:
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
                else:
                    raise ValueError(f"unexpected metric {metric_name}")

                xs, ys = __resample(data)

                if len(metrics) > 1:
                    if len(c_values) == 1:
                        plot_label = f"{__legend(metric_name)}"
                    else:
                        plot_label = f"{__legend(metric_name)} $c={c}$"
                else:
                    plot_label = f"$c={c}$"

                plot_color = (
                    coloring_rules[f"c={c}"]
                    or coloring_rules[f"m={metric_name}"]
                    or DEFAULT_LINECOLOR
                )
                plot_linestyle = (
                    styling_rules[f"c={c}"]
                    or styling_rules[f"m={metric_name}"]
                    or DEFAULT_LINESTYLE
                )

                (axr if metric_name in metrics_right else axl).plot(
                    xs,
                    ys,
                    label=plot_label,
                    color=plot_color,
                    linestyle=plot_linestyle,
                )

                if metric_name in metrics_right:
                    axl.plot(
                        [],
                        [],
                        color=plot_color,
                        linestyle=plot_linestyle,
                        label=plot_label,
                    )

    if (len(c_values) * len(metrics)) > 1:
        if "ge_bar" in metrics:
            axl.legend(loc="lower right")
        else:
            axl.legend(loc="upper right")

    return fig


def save_fig(
    fig: Figure,
    fname: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    bbox_inches: str = "tight",
    pad_inches: float = 0,
    dpi: int = 600,
) -> None:
    if xlim is not None:
        for ax in fig.axes:
            ax.set_xlim(*xlim)
    if ylim is not None:
        for ax in fig.axes:
            ax.set_ylim(*ylim)
    fig.savefig(fname, bbox_inches=bbox_inches, pad_inches=pad_inches, dpi=dpi)
