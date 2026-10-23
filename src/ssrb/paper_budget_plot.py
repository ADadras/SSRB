"""Publication figure layout, with explicit portable data inputs."""

from pathlib import Path
import csv,json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

METRICS = [f"MLQ@{k}" for k in range(1, 9)]

METHODS = ["SSRB", "Probabilistic U-Net", "DINO Capacity", "Raw Capacity"]

STYLE = {
    "SSRB": dict(color="#009988", marker="D", linestyle="-", linewidth=1.65),
    "Probabilistic U-Net": dict(color="#0077BB", marker="o", linestyle="--", linewidth=1.35),
    "DINO Capacity": dict(color="#6F7680", marker="^", linestyle=":", linewidth=1.25),
    "Raw Capacity": dict(color="#EE7733", marker="s", linestyle="-.", linewidth=1.25),
}

def add_axis_break(ax):
    kwargs = dict(transform=ax.transAxes, color="#555555", clip_on=False, linewidth=.75)
    ax.plot((-.015, .005), (-.018, .018), **kwargs)
    ax.plot((.012, .032), (-.018, .018), **kwargs)

def method_curves(source):
    values=json.loads(Path(source).read_text())
    if set(values)!={'LIDC','LIDC-hard','IMA++'}:
        raise ValueError('requires LIDC, LIDC-hard and IMA++ curves')
    for regime,methods in values.items():
        if set(methods)!=set(METHODS):
            raise ValueError('requires all four comparison methods')
        for method,curve in methods.items():
            array=np.asarray(curve,dtype=np.float64)
            if array.shape!=(8,) or not np.isfinite(array).all() or (array<0).any() or (array>1).any():
                raise ValueError('requires eight finite prefix coverage values')
            if (np.diff(array)<-1e-7).any():
                raise ValueError('prefix coverage must be nondecreasing')
            methods[method]=array
    return values


def main(source, output) -> None:
    curves = method_curves(source)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": .025,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    # The taller canvas uses the available WACV page space and keeps all labels
    # readable after the figure is reduced to double-column width.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.35), sharex=True, sharey=True)
    ks = np.arange(1, 9)
    titles = ["(a) LIDC", "(b) LIDC-hard", "(c) IMA++"]

    for ax, regime, title in zip(axes, ["LIDC", "LIDC-hard", "IMA++"], titles):
        for method in METHODS:
            style = STYLE[method]
            ax.plot(
                ks,
                curves[regime][method],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=3.7,
                markeredgecolor="white",
                markeredgewidth=.35,
                zorder=4 if method == "SSRB" else 3,
            )
        ax.set_title(title, loc="left", weight="bold", pad=4)
        ax.set_xlim(.72, 8.28)
        ax.set_ylim(.65, .948)
        ax.set_xticks(ks)
        ax.set_yticks([.70, .75, .80, .85, .90])
        ax.grid(axis="y", color="#E1E5E7", linewidth=.5)
        ax.set_xlabel("bank budget $k$")
        add_axis_break(ax)

        ssrb = curves[regime]["SSRB"]
        prob = curves[regime]["Probabilistic U-Net"]
        first_effect = ssrb[0] - prob[0]
        last_effect = ssrb[-1] - prob[-1]
        # Put the endpoint comparisons in a clear top strip instead of drawing
        # callouts through the curves. The caption states the k=3 crossover.
        if first_effect < 0:
            first_text = rf"$k=1$: Prob-U +{abs(first_effect):.3f}"
            first_color = STYLE["Probabilistic U-Net"]["color"]
        else:
            first_text = rf"$k=1$: SSRB +{first_effect:.3f}"
            first_color = STYLE["SSRB"]["color"]
        label_box = dict(facecolor="white", edgecolor="none", alpha=.92, pad=.7)
        ax.text(
            .025,
            .965,
            first_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=first_color,
            bbox=label_box,
            zorder=8,
        )
        ax.text(
            .025,
            .895,
            rf"$k=8$: SSRB +{last_effect:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=STYLE["SSRB"]["color"],
            bbox=label_box,
            zorder=8,
        )

    axes[0].set_ylabel("MLQ@$k$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(.5, .995),
        handlelength=2.0,
        handletextpad=.45,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=.072, right=.995, bottom=.16, top=.82, wspace=.10)

    output=Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
