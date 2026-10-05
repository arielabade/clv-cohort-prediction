"""Report figures, built from the committed report files.

Chart rule, unchanged from the first version of this file: neutrals carry the
structure and a single accent carries the finding. What changed is the
surface. These figures now sit on the same ``#050505`` panel as the rest of
the portfolio, so a README reads as one object rather than as a page with
screenshots pasted into it.

Run with ``python -m clv.figures``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import brandviz as bv

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "assets" / "figures"


def cohort_heatmap(matrix: pd.DataFrame) -> Path:
    """Monthly retention by acquisition cohort.

    A heatmap rather than twenty-odd lines: the question is whether later
    cohorts retain worse than earlier ones, which is a question about the
    shape of a block, not about any one cohort's path.
    """
    # Month 0 is 100% for every cohort by construction. Keeping it would spend
    # the lightest end of the ramp on a column that carries no information.
    data = matrix.iloc[:, 1:14] * 100

    fig, ax = bv.panel(
        12.4, 7.2,
        title="Retention by acquisition cohort",
        subtitle="Share of each monthly cohort still purchasing, by months since first order",
    )
    image = ax.imshow(data.to_numpy(), cmap=bv.SEQUENTIAL, aspect="auto", vmin=0, vmax=45)
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(len(data)), data.index)
    ax.set_xlabel("Months since acquisition")

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data.iat[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7.2,
                        color=bv.CARBON if value > 38 else bv.IVORY)

    bar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.015)
    bar.set_label("% of cohort still active", color=bv.STEEL, fontsize=9.5)
    bar.ax.tick_params(colors=bv.STEEL)
    bar.outline.set_edgecolor(bv.GRID)
    ax.spines[["top", "right", "bottom", "left"]].set_color(bv.GRID)
    ax.tick_params(length=0)
    return bv.save(fig, FIGURES / "cohort_retention.svg")


def decile_lift(bgnbd: pd.DataFrame, lightgbm: pd.DataFrame) -> Path:
    """Does the ranking hold up on customers neither model has seen?

    Two peer models, so the categorical palette rather than accent-and-grey,
    and both series carry a legend so the comparison never rests on telling
    indigo from amber.
    """
    fig, ax = bv.panel(
        12.4, 5.6,
        title="Both models rank, but only one ranks the top decile right",
        subtitle="Actual holdout value of each predicted decile, against the average customer",
    )
    ax.set_xlim(-0.6, len(bgnbd) - 0.4)
    ax.set_ylim(0, max(bgnbd["lift_vs_average"].max(), lightgbm["lift_vs_average"].max()) * 1.26)
    fig.canvas.draw()

    bv.grouped_bars(
        ax, [str(decile) for decile in bgnbd["decile"]],
        {"BG/NBD + Gamma-Gamma": bgnbd["lift_vs_average"],
         "LightGBM": lightgbm["lift_vs_average"]},
        colors=(bv.COBALT, bv.AMBER),
    )
    ax.set_xlabel("Predicted-value decile (1 = highest predicted)")
    ax.set_ylabel("Actual value vs the average customer")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}x")
    bv.reference_line(ax, 1.0, "average customer", where=0.985)
    ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1.0, 1.06))

    bv.annotate(
        ax,
        f"Top decile: {bgnbd['lift_vs_average'].iloc[0]:.1f}x against "
        f"{lightgbm['lift_vs_average'].iloc[0]:.1f}x.\nThis is the decile a retention budget is actually spent on",
        xy=(0.26, bgnbd["lift_vs_average"].iloc[0] * 0.99),
        xytext=(1.6, ax.get_ylim()[1] * 0.72), color=bv.IVORY,
    )
    bv.clean(ax)
    return bv.save(fig, FIGURES / "decile_lift.svg")


def value_capture(bgnbd: pd.DataFrame, lightgbm: pd.DataFrame) -> Path:
    """How much future value you reach by contacting the top N deciles.

    This is the chart a budget is argued from: the y value at x=10% is the
    share of all future revenue a campaign of that size can touch.
    """
    fig, ax = bv.panel(
        12.4, 5.4,
        title="Share of future value reached, by how much of the base you contact",
        subtitle="Cumulative actual holdout value of the top predicted deciles, best first",
    )
    share = np.arange(1, len(bgnbd) + 1) * 10

    for frame, colour, label in (
        (bgnbd, bv.COBALT, "BG/NBD + Gamma-Gamma"),
        (lightgbm, bv.AMBER, "LightGBM"),
    ):
        captured = frame["share_of_actual_value"].cumsum().to_numpy() * 100
        ax.plot(np.concatenate([[0], share]), np.concatenate([[0], captured]),
                color=colour, linewidth=2.4, marker="o", markersize=8,
                markeredgecolor=bv.CARBON, markeredgewidth=1.8, zorder=4, label=label)
        ax.text(share[0] + 1.6, captured[0], f"{captured[0]:.1f}%", color=bv.IVORY,
                fontsize=10, fontweight="bold", va="center")

    ax.plot([0, 100], [0, 100], color=bv.SLATE, linewidth=2.0,
            linestyle=(0, (5, 4)), zorder=2, label="No ranking at all")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of the customer base contacted, best first")
    ax.set_ylabel("Share of future value reached")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(loc="lower right")
    bv.clean(ax, axis="both", spines=("top", "right"))
    return bv.save(fig, FIGURES / "value_capture.svg")


def headline(bgnbd: pd.DataFrame, metrics: dict):
    """The three numbers the README leads with."""
    fig, _ = bv.kpi_strip([
        (f"{bgnbd['share_of_actual_value'].iloc[0]:.1%}",
         "Of all future value sits in the\ntop predicted decile of customers"),
        (f"£{metrics['bgnbd']['mae']:.0f} vs £{metrics['lightgbm']['mae']:.0f}",
         "Mean absolute error, probabilistic\nmodel against gradient boosting"),
        (f"{metrics['bgnbd']['spearman']:.3f}",
         f"Rank correlation on a {metrics['horizon_days']}-day holdout\n"
         f"({metrics['lightgbm']['spearman']:.3f} for LightGBM)"),
    ])
    return bv.save(fig, FIGURES / "headline.svg")


def build_all() -> list[Path]:
    matrix = pd.read_csv(REPORTS / "cohort_retention.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)
    bgnbd = pd.read_csv(REPORTS / "deciles_bgnbd.csv")
    lightgbm = pd.read_csv(REPORTS / "deciles_lightgbm.csv")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    return [
        headline(bgnbd, metrics),
        decile_lift(bgnbd, lightgbm),
        value_capture(bgnbd, lightgbm),
        cohort_heatmap(matrix),
    ]


if __name__ == "__main__":
    for path in build_all():
        print(path.relative_to(ROOT))
