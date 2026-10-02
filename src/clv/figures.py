"""Report figures.

Chart rule: neutrals carry the structure and a single accent carries the
finding. No rainbow without semantics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CARBON, GRAPHITE, STEEL, COBALT, IVORY = "#050505", "#1B1C1F", "#7E8791", "#5B6CFF", "#F6F5F0"
HAIRLINE = "#D8D9DC"

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _style() -> None:
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    stack = [n for n in ("Lato", "Helvetica Neue", "Arial") if n in available] + ["DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": stack,
            "axes.edgecolor": HAIRLINE,
            "axes.labelcolor": GRAPHITE,
            "axes.titlecolor": CARBON,
            "axes.titleweight": "bold",
            "text.color": GRAPHITE,
            "xtick.color": STEEL,
            "ytick.color": STEEL,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def cohort_heatmap(matrix: pd.DataFrame, path: Path | None = None) -> Path:
    _style()
    data = matrix.iloc[:, :13] * 100
    cmap = LinearSegmentedColormap.from_list("cobalt", [IVORY, COBALT])

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(data.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=45)
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(len(data)), data.index)
    ax.set_xlabel("Months since acquisition")
    ax.set_title("Monthly retention by acquisition cohort (%)", pad=14)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data.iat[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7,
                        color=IVORY if value > 28 else GRAPHITE)
    fig.colorbar(image, ax=ax, shrink=0.7, label="% of cohort active")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = path or REPORTS / "cohort_retention.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def decile_lift(bgnbd: pd.DataFrame, lightgbm: pd.DataFrame, path: Path | None = None) -> Path:
    """Lift by predicted decile.

    The probabilistic model is the accent because it is the one that answers
    the question; the comparison sits in neutral.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.4
    positions = range(len(bgnbd))
    ax.bar([p - width / 2 for p in positions], bgnbd["lift_vs_average"], width,
           color=COBALT, label="BG/NBD + Gamma-Gamma")
    ax.bar([p + width / 2 for p in positions], lightgbm["lift_vs_average"], width,
           color=STEEL, alpha=0.55, label="LightGBM")
    ax.axhline(1.0, color=GRAPHITE, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(len(bgnbd) - 0.4, 1.06, "average customer", fontsize=8, color=STEEL, ha="right")

    ax.set_xticks(list(positions), [str(d) for d in bgnbd["decile"]])
    ax.set_xlabel("Predicted-value decile (1 = highest)")
    ax.set_ylabel("Actual holdout value vs average")
    ax.set_title("Does the ranking hold up on unseen customers?", pad=14)
    ax.legend(frameon=False)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    path = path or REPORTS / "decile_lift.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    matrix = pd.read_csv(REPORTS / "cohort_retention.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)
    print(cohort_heatmap(matrix))
    print(decile_lift(
        pd.read_csv(REPORTS / "deciles_bgnbd.csv"),
        pd.read_csv(REPORTS / "deciles_lightgbm.csv"),
    ))
