from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Callable, Generator, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ferm_ge import ExpTrainResult, ExpValidResult, ParamSet

__all__ = [
    "PlottingData",
    "make_plottingdata",
    "plot_results",
    "save_fig",
    "parse_metric",
]

DEFAULT_FIGSIZE = (3.0, 1.5)
DEFAULT_COLOR = "black"
DEFAULT_LINESTYLE = "solid"

LR = Literal["left", "right"]
TV = Literal["train", "valid"]
FArrayLike = NDArray[np.float_] | list[float] | tuple[float, ...] | None


@dataclass
class PlottingData:
    name: str

    title: str | None
    xlabel: str | None
    ylabel: str | None
    legend_loc: str
    figsize: tuple[float, float]

    y: dict[str, FArrayLike]
    x: defaultdict[str, FArrayLike]
    err: defaultdict[str, FArrayLike]

    base: defaultdict[str, list[float] | None]

    axis: defaultdict[str, LR]
    color: defaultdict[str, str]
    linestyle: defaultdict[str, str]
    linewidth: defaultdict[str, float]
    legend: defaultdict[str, str | None]

    def get_dict(self):
        return {
            "name": self.name,
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "legend_loc": self.legend_loc,
            "figsize": self.figsize,
            "y": self.y,
            "x": self.x,
            "err": self.err,
            "base": self.base,
            "axis": self.axis,
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linestyle,
            "legend": self.legend,
        }

    @staticmethod
    def new() -> "PlottingData":
        return PlottingData(
            name="unknown",
            title=None,
            xlabel=None,
            ylabel=None,
            legend_loc="best",
            figsize=DEFAULT_FIGSIZE,
            y={},
            x=defaultdict[str, FArrayLike](lambda: None),
            err=defaultdict[str, FArrayLike](lambda: None),
            base=defaultdict[str, list[float] | None](lambda: None),
            axis=defaultdict[str, LR](lambda: "left"),
            color=defaultdict[str, str](lambda: DEFAULT_COLOR),
            linestyle=defaultdict[str, str](lambda: DEFAULT_LINESTYLE),
            linewidth=defaultdict[str, float](lambda: 1.5),
            legend=defaultdict[str, str | None](lambda: None),
        )


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


def parse_metric(
    metric: str,
) -> tuple[
    TV,
    str,
    str | None,
    str | None,
    LR,
    str,
    str,
    str | None,
    str,
    float,
    Callable[[ParamSet], bool],
]:
    target: TV
    if metric.startswith("t:"):
        target = "train"
    elif metric.startswith("v:"):
        target = "valid"
    else:
        raise ValueError("metric must start with t: or v:")

    metric_items = metric[2:].split(":")
    exp_val = metric_items[0]

    exp_err = None
    exp_base = None
    axis: LR = "left"
    color = DEFAULT_COLOR
    linestyle = DEFAULT_LINESTYLE
    linewidth = 1.5
    legend = None
    name = None
    filts: list[str] = []
    for metric_item in metric_items[1:]:
        if metric_item.startswith("e!"):
            exp_err = metric_item[2:]
        elif metric_item.startswith("b!"):
            exp_base = metric_item[2:]
        elif metric_item.startswith("c!"):
            color = metric_item[2:]
        elif metric_item.startswith("s!"):
            linestyle = metric_item[2:]
        elif metric_item.startswith("l!"):
            legend = metric_item[2:]
        elif metric_item.startswith("a!"):
            assert metric_item[2:] in [
                "left",
                "right",
            ], "axis must be left or right"
            axis = metric_item[2:]  # type: ignore
        elif metric_item.startswith("n!"):
            name = metric_item[2:]
        elif metric_item.startswith("f!"):
            filts = metric_item[2:].split(",")
        elif metric_item.startswith("w!"):
            linewidth = float(metric_item[2:])
        else:
            raise ValueError(f"unknown metric item: {metric_item}")

    if name is None:
        name = re.sub(r"\W+", "", exp_val)

    def filt(ps: ParamSet) -> bool:
        return all([bool(eval(f, {"p": ps})) for f in filts])

    return (
        target,
        exp_val,
        exp_err,
        exp_base,
        axis,
        color,
        linestyle,
        legend,
        name,
        linewidth,
        filt,
    )


def parse_metricopt(
    metricopt: str,
) -> tuple[tuple[float, float], str | None, str | None, str | None, str]:
    figsize = DEFAULT_FIGSIZE
    xlabel = None
    ylabel = None
    title = None
    legend_loc = "best"

    assert metricopt.startswith("fig:"), "metricopt must start with fig:"
    metricopt_items = metricopt.split(":")
    for metricopt_item in metricopt_items[1:]:
        if metricopt_item.startswith("f!"):
            figsize = tuple(map(float, metricopt_item[2:].split(",")))  # type: ignore
        elif metricopt_item.startswith("x!"):
            xlabel = metricopt_item[2:]
        elif metricopt_item.startswith("y!"):
            ylabel = metricopt_item[2:]
        elif metricopt_item.startswith("t!"):
            title = metricopt_item[2:]
        elif metricopt_item.startswith("l!"):
            legend_loc = metricopt_item[2:]
        else:
            raise ValueError(f"unknown metricopt item: {metricopt_item}")

    return figsize, xlabel, ylabel, title, legend_loc


def make_plottingdata(
    results: dict[ParamSet, tuple[ExpTrainResult, ExpValidResult | None]],
    metrics: list[str],
    confidence: float = 0.0,
    x_axis: str | None = None,
    figsep_rules: list[str] = ["p.alpha"],
) -> Generator[PlottingData, None, None]:
    assert (
        confidence >= 0 and confidence < 1
    ), "confidence value must be in [0, 1)"
    z_value = statistics.NormalDist().inv_cdf((1 + confidence) / 2)

    # metrics_left and metrics_right are lists of metric strings that are
    # plotted on the left and right y-axis, respectively.
    # Metric strings are formatted as follows:
    #   "{t,v}:EXPR1[:{e,b,c,s,l,a,n,f,w}!EXPR2]..."
    #     - t: train results, v: valid results
    #     - EXPR1: to be evaluated or just the name of the attribute
    #     - e!: use the given expression as confidence band (stddev)
    #     - b!: use the given expression as baseline
    #     - c!: use the given color (optional; default: black)
    #     - s!: use the given linestyle (optional; default: solid)
    #     - l!: use the given legend (optional; default: None)
    #     - a!: use the given axis (optional; default: left)
    #     - n!: use the given name (optional; default: extracted from EXPR1)
    #     - f!: use the given filter (optional; default: no filtering)
    #     - w!: use the given linewidth (optional; default: 1.5)
    assert len(metrics) > 0, "metrics must not be empty"
    use_train_results = False
    use_valid_results = False
    figsize = DEFAULT_FIGSIZE
    xlabel = None
    ylabel = None
    title = None
    legend_loc = "best"
    metrics4plot = []
    for metric in metrics:
        assert metric.count(":") >= 1, f"invalid metric: {metric}"
        if metric.startswith("t:"):
            assert x_axis is None, "x_axis cannot be used with train results"
            use_train_results = True
            metrics4plot.append(metric)
        elif metric.startswith("v:"):
            assert (
                x_axis is not None and x_axis[:2] == "p."
            ), "x_axis must be specified using parameters with valid results"
            use_valid_results = True
            metrics4plot.append(metric)
        elif metric.startswith("fig:"):
            figsize, xlabel, ylabel, title, legend_loc = parse_metricopt(metric)
        else:
            raise ValueError(f"unknown metric: {metric}")
    assert not (
        use_train_results and use_valid_results
    ), "cannot use both train and valid results"

    # figsep_rules is a list of parameter names that are used to separate
    # figures. For example, if figsep_rules = ["p.lambda_max", "p.nu"],
    # then the figures will be separated by lambda_max and nu.
    figsep_baselist: list[set[tuple[str, float]]] = []
    for rule_str in figsep_rules:
        assert rule_str.startswith("p."), "rule must start with p."
        rule_name = rule_str[2:]
        assert (
            rule_name in ParamSet.__annotations__
        ), f"unknown parameter rule: {rule_name}"
        figsep_base = []
        for ps in results:
            figsep_base.append((rule_name, float(getattr(ps, rule_name))))
        figsep_baselist.append(set(figsep_base))

    figsep: tuple[tuple[str, float]]
    for figsep in product(*figsep_baselist):  # type: ignore
        ps_list: list[ParamSet] = []
        for ps in results:
            use_this = True
            for rule_p, rule_pv in figsep:
                if getattr(ps, rule_p) != rule_pv:
                    use_this = False
                    break
            if use_this:
                ps_list.append(ps)
        if len(ps_list) == 0:
            continue

        # now ps_list contains all ParamSet objects that should be plotted in
        # the same figure. from now, we will gather data with given metrics,
        # and find proper legend/axis/color/linestyle for each metric.

        data = PlottingData.new()
        data.figsize = figsize
        data.xlabel = xlabel
        data.ylabel = ylabel
        data.title = title
        data.legend_loc = legend_loc
        data_names = []

        if use_train_results:
            for ps in ps_list:
                for metric in metrics4plot:
                    (
                        _,
                        exp,
                        _,
                        _,
                        axis,
                        color,
                        linestyle,
                        legend_label,
                        name,
                        linewidth,
                        filt,
                    ) = parse_metric(metric)
                    if not filt(ps):
                        continue
                    data_names.append(name)

                    part_id = f"{ps}_{metric}"

                    data.axis[part_id] = axis
                    data.color[part_id] = color
                    data.linestyle[part_id] = linestyle
                    data.linewidth[part_id] = linewidth

                    if legend_label is not None:
                        data.legend[part_id] = legend_label.format(
                            **ps.__dict__
                        )

                    data.x[part_id], data.y[part_id] = __resample(
                        np.array(
                            eval(
                                exp,
                                results[ps][0].__dict__,
                            )
                        ).astype(np.float_)
                    )

        if use_valid_results:
            assert x_axis is not None, "x_axis must be specified"
            for metric in metrics4plot:
                part_id = metric
                (
                    _,
                    exp_val,
                    exp_err,
                    exp_base,
                    axis,
                    color,
                    linestyle,
                    legend_label,
                    name,
                    linewidth,
                    filt,
                ) = parse_metric(metric)

                ps_list_filtered = []
                for ps in ps_list:
                    if filt(ps):
                        ps_list_filtered.append(ps)

                if len(ps_list_filtered) == 0:
                    continue
                data_names.append(name)

                data.axis[part_id] = axis
                data.color[part_id] = color
                data.linestyle[part_id] = linestyle
                data.linewidth[part_id] = linewidth

                if exp_err is not None:
                    err_list = []
                    for ps in ps_list_filtered:
                        err_list.append(
                            float(eval(exp_err, results[ps][1].__dict__))
                            * z_value
                        )
                    data.err[part_id] = err_list

                if exp_base is not None:
                    base_list = []
                    for ps in ps_list_filtered:
                        base_list.append(
                            float(eval(exp_base, results[ps][1].__dict__))
                        )
                    data.base[part_id] = list(set(base_list))

                if legend_label is not None:
                    data.legend[part_id] = legend_label.format(
                        **ps_list_filtered[0].__dict__
                    )

                x_list = []
                y_list = []
                for ps in ps_list_filtered:
                    x = float(getattr(ps, x_axis[2:]))
                    y = float(eval(exp_val, results[ps][1].__dict__))
                    x_list.append(x)
                    y_list.append(y)
                data.x[part_id] = x_list
                data.y[part_id] = y_list

        data.name = "-".join(sorted(list(set(data_names))))

        yield data


def plot_results(
    data: PlottingData,
    ypad: float | None = None,
    xpad: float | None = None,
    ylim: list[tuple[float, float] | None] | None = None,
    xlim: tuple[float, float] | None = None,
    use_tex: bool = False,
) -> Figure:
    if use_tex:
        plt.rcParams["ps.useafm"] = True
        plt.rcParams["pdf.use14corefonts"] = True
        plt.rcParams["text.usetex"] = True
    else:
        plt.rcParams["ps.useafm"] = False
        plt.rcParams["pdf.use14corefonts"] = False
        plt.rcParams["text.usetex"] = False

    keys = data.y.keys()

    use_axr = any([data.axis[k] == "right" for k in keys])
    use_axl = any([data.axis[k] == "left" for k in keys])
    use_legend = any([data.legend[k] is not None for k in keys])

    fig, axl = plt.subplots(figsize=data.figsize)
    axr = axl.twinx() if use_axr else axl

    xmin, xmax = float("inf"), float("-inf")
    ymin: dict[LR, float] = {"left": float("inf"), "right": float("inf")}
    ymax: dict[LR, float] = {"left": float("-inf"), "right": float("-inf")}
    baselines: list[tuple[str, float, Axes]] = []

    for k in keys:
        ax = axr if data.axis[k] == "right" else axl
        color = data.color[k]
        linestyle = data.linestyle[k]
        linewidth = data.linewidth[k]
        legend = data.legend[k]

        y = np.array(data.y[k]).astype(np.float_)
        x = (
            np.array(data.x[k]) if data.x[k] is not None else np.arange(len(y))
        ).astype(np.float_)

        if data.err[k] is not None:
            err = np.array(data.err[k]).astype(np.float_)
            upper = y + err
            lower = y - err
            ax.fill_between(
                x,
                lower,  # type: ignore
                upper,  # type: ignore
                color=color,
                alpha=0.2,
            )

            ymin[data.axis[k]] = min(ymin[data.axis[k]], lower.min())
            ymax[data.axis[k]] = max(ymax[data.axis[k]], upper.max())

        ymin[data.axis[k]] = min(ymin[data.axis[k]], y.min())
        ymax[data.axis[k]] = max(ymax[data.axis[k]], y.max())
        xmin = min(xmin, x.min())
        xmax = max(xmax, x.max())

        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        if use_legend:
            axr.plot(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=legend,
            )

        baseline = data.base[k]
        if baseline is not None:
            for b in baseline:
                baselines.append((color, b, ax))

    # draw baselines
    for color, y_value, ax in baselines:
        ax.axhline(y_value, color=color, linestyle="--", alpha=0.4)

    # set xlim and ylim
    assert not (
        xlim is not None and xpad is not None
    ), "xlim and xpad cannot be set at the same time"
    if xlim is not None:
        axl.set_xlim(*xlim)
    elif xmin != xmax and xpad is not None:
        xpad_value = xpad * (xmax - xmin) / (1 - 2 * xpad)
        axl.set_xlim(xmin - xpad_value, xmax + xpad_value)

    assert not (
        ylim is not None and ypad is not None
    ), "ylim and ypad cannot be set at the same time"
    if ylim is not None:
        if use_axl:
            assert len(ylim) > 0, "ylim must have at least one element"
            assert ylim[0] is not None, "ylim[0] cannot be None"
            axl.set_ylim(*ylim[0])
        if use_axr:
            assert len(ylim) > 1, "ylim must have at least two elements"
            assert ylim[1] is not None, "ylim[1] cannot be None"
            axr.set_ylim(*ylim[1])
    elif ypad is not None:
        if use_axl:
            lymin, lymax = ymin["left"], ymax["left"]
            lypad_value = ypad * (lymax - lymin) / (1 - 2 * ypad)
            axl.set_ylim(lymin - lypad_value, lymax + lypad_value)
        if use_axr:
            rymin, rymax = ymin["right"], ymax["right"]
            rypad_value = ypad * (rymax - rymin) / (1 - 2 * ypad)
            axr.set_ylim(rymin - rypad_value, rymax + rypad_value)

    # draw title
    if data.title is not None:
        axl.set_title(data.title)

    # draw labels
    if data.xlabel is not None:
        axl.set_xlabel(data.xlabel)
    if data.ylabel is not None:
        axl.set_ylabel(data.ylabel)

    # draw legend
    if use_legend:
        axr.legend(loc=data.legend_loc)

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
