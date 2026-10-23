"""Publication figure layout, with explicit portable data inputs."""

from pathlib import Path
import csv,json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ENDPOINTS = ["LIDC-hard", "BSDS", "IMA++", "RIGA-disc", "RIGA-cup"]

METHODS = ["DINO Capacity-WTA", "Raw Capacity-WTA", "SSRB (ours)"]

STYLE = {
    "DINO Capacity-WTA": dict(label="DINO Capacity", color="#0077BB", marker="o"),
    "Raw Capacity-WTA": dict(label="Raw Capacity", color="#EE7733", marker="s"),
    "SSRB (ours)": dict(label="SSRB", color="#009988", marker="D"),
}

def read_values(source) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with Path(source).open(newline="") as stream:
        for row in csv.DictReader(stream):
            endpoint = row["regime"]
            method = row["method"]
            if endpoint in ENDPOINTS and method in METHODS:
                values[(endpoint, method)] = float(row["MLQ8"])
    expected = {(endpoint, method) for endpoint in ENDPOINTS for method in METHODS}
    if set(values) != expected:
        raise ValueError("the branch table does not contain the required 5 x 3 values")
    return values

def main(source, output) -> None:
    values = read_values(source)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": .025,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.16, 2.42))

    # The first two rows are the extent/structure contrast emphasized in the text.
    ax.axhspan(2.52, 4.48, color="#EAF2F6", zorder=0)
    ax.axhline(2.48, color="#B8C5CC", linewidth=.65, zorder=1)

    base_y = {endpoint: 4 - index for index, endpoint in enumerate(ENDPOINTS)}
    offsets = {
        "DINO Capacity-WTA": .20,
        "Raw Capacity-WTA": 0.0,
        "SSRB (ours)": -.20,
    }

    # A faint range line helps the eye compare the three predictions within a task.
    for endpoint in ENDPOINTS:
        xs = [values[(endpoint, method)] for method in METHODS]
        ax.plot([min(xs), max(xs)], [base_y[endpoint], base_y[endpoint]],
                color="#C9D0D4", linewidth=.75, zorder=1)

    for method in METHODS:
        style = STYLE[method]
        xs = [values[(endpoint, method)] for endpoint in ENDPOINTS]
        ys = [base_y[endpoint] + offsets[method] for endpoint in ENDPOINTS]
        ax.scatter(
            xs,
            ys,
            s=27 if method == "SSRB (ours)" else 23,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="white",
            linewidth=.65,
            label=style["label"],
            zorder=4,
        )
        if method == "SSRB (ours)":
            for endpoint, x, y in zip(ENDPOINTS, xs, ys):
                ax.text(x + .008, y, f"{x:.3f}", va="center", ha="left",
                        fontsize=6.15, color=style["color"], weight="bold")

    ax.set_xlim(.38, 1.015)
    ax.set_ylim(-.47, 4.47)
    ax.set_xticks([.4, .5, .6, .7, .8, .9, 1.0])
    ax.set_xlabel("MLQ@8 (higher is better)")
    ax.set_yticks([base_y[endpoint] for endpoint in ENDPOINTS])
    labels = ax.set_yticklabels(ENDPOINTS)
    for index, label in enumerate(labels):
        if index < 2:
            label.set_weight("bold")
            label.set_color("#324B5A")
    ax.grid(axis="x", color="#E2E6E8", linewidth=.55, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#7C8A91")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(.56, 1.115),
        handletextpad=.35,
        columnspacing=1.35,
    )
    ax.text(.385, 4.43, "Complementary branch strengths", fontsize=6.25,
            color="#526C7A", ha="left", va="top")

    output=Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
