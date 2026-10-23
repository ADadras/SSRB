"""Publication figure layout, with explicit portable data inputs."""

from pathlib import Path
import csv,json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "LIDC": "#3B6FB6",
    "LIDC-hard": "#234A7C",
    "IMA++": "#D17A22",
    "RIGA-disc": "#2E8B57",
    "RIGA-cup": "#72A84A",
}

def main(source, output) -> None:
    effects = pd.read_csv(source)
    prefix = effects[effects.metric.isin([f"MLQ@{k}" for k in range(1,9)])].copy()
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    for regime, color in COLORS.items():
        group = prefix[prefix.regime == regime].copy()
        group["k"] = group.metric.str.replace("MLQ@", "", regex=False).astype(int)
        group = group.sort_values("k")
        ax.plot(group.k, group.effect, marker="o", markersize=3.0, linewidth=1.25,
                color=color, label=regime)
        ax.fill_between(group.k, group.ci95_low, group.ci95_high, color=color, alpha=0.10, linewidth=0)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("returned-bank budget $k$")
    ax.set_ylabel("SSRB $-$ Prob-U MLQ@$k$")
    ax.set_title("(a) Coverage gain emerges with bank budget", loc="left", fontsize=7.6, weight="bold")
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=6.2, loc="lower right", columnspacing=0.8)

    ax = axes[1]
    rows = effects[effects.metric.isin(["MLQ@8", "Waste@8"])].pivot(index="regime", columns="metric")
    offsets = {
        "LIDC": (3, -9), "LIDC-hard": (3, 5), "IMA++": (-35, -11),
        "RIGA-disc": (3, -10), "RIGA-cup": (3, 4),
    }
    for regime, color in COLORS.items():
        x = float(rows.loc[regime, ("effect", "Waste@8")])
        y = float(rows.loc[regime, ("effect", "MLQ@8")])
        xlo = float(rows.loc[regime, ("simultaneous_ci95_low", "Waste@8")])
        xhi = float(rows.loc[regime, ("simultaneous_ci95_high", "Waste@8")])
        ylo = float(rows.loc[regime, ("simultaneous_ci95_low", "MLQ@8")])
        yhi = float(rows.loc[regime, ("simultaneous_ci95_high", "MLQ@8")])
        ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                    fmt="o", markersize=4.2, capsize=2, linewidth=0.8, color=color)
        ax.annotate(regime, (x, y), xytext=offsets[regime], textcoords="offset points",
                    fontsize=6.1, color=color)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.axvline(0, color="#555555", linewidth=0.7)
    ax.set_xlabel("SSRB $-$ Prob-U Waste@8")
    ax.set_ylabel("SSRB $-$ Prob-U MLQ@8")
    ax.set_title("(b) K=8 coverage–Waste tradeoff", loc="left", fontsize=7.6, weight="bold")
    ax.grid(color="#E3E3E3", linewidth=0.5)

    fig.tight_layout(w_pad=1.4)
    output=Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
