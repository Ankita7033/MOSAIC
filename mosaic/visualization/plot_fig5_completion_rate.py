#!/usr/bin/env python3
"""
Fig. 5 — Mission-Aware Completion Rate by Workload Class
=========================================================
Compares PriorityStrict vs MOSAIC under crisis-scenario burst.

MOSAIC achieves 100% completion for all three mission-critical classes:
  inference (YOLO), dispatch (FastAPI), sensor_fusion (IoT Sync).
Low-priority classes are gracefully deferred:
  analytics 42.9%, model_update 14.3%.
PriorityStrict achieves 85.7% on inference but provides no interference isolation.

Usage:
    python plot_fig5_completion_rate.py
    python plot_fig5_completion_rate.py --output results/fig5_completion_rate.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    raise SystemExit(1)

_ROOT = Path(__file__).parent.parent


def generate_fig5(output_path: Path | None = None):
    """Generate Fig. 5: Mission-aware completion rate by workload class."""

    # ── Data ──────────────────────────────────────────────────────────────
    classes = [
        "Inference\n(YOLO)",
        "Dispatch\n(FastAPI)",
        "Sensor Fusion\n(IoT Sync)",
        "Analytics\n(Batch)",
        "Model Update\n(Edge FL)",
    ]
    # Mission-critical classes first, then deferred
    mosaic_rates     = [100.0, 100.0, 100.0, 42.9, 14.3]
    priority_rates   = [85.7,  76.2,  77.8,  50.0, 100.0]
    n_classes        = len(classes)
    mission_critical = 3  # first 3 are mission-critical

    # ── Style ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Background shading ────────────────────────────────────────────────
    # Mission-critical region (light green)
    ax.axhspan(-0.5, mission_critical - 0.5, color="#e6f9e6", zorder=0)
    # Deferred region (light amber)
    ax.axhspan(mission_critical - 0.5, n_classes - 0.5, color="#fff8e1", zorder=0)

    # Region labels
    ax.text(103, (mission_critical - 1) / 2, "Mission-\nCritical",
            ha="left", va="center", fontsize=9, fontstyle="italic",
            color="#2e7d32", fontweight="bold")
    ax.text(103, mission_critical + 0.5, "Deferred",
            ha="left", va="center", fontsize=9, fontstyle="italic",
            color="#f57f17", fontweight="bold")

    # ── Bar positions ─────────────────────────────────────────────────────
    bar_height = 0.35
    y_positions = list(range(n_classes))

    # PriorityStrict bars (upper, cyan)
    y_priority = [y - bar_height / 2 for y in y_positions]
    bars_p = ax.barh(y_priority, priority_rates, height=bar_height,
                     color="#4dd0e1", edgecolor="#00838f", linewidth=0.6,
                     label="PriorityStrict", zorder=3)

    # MOSAIC bars (lower, green)
    y_mosaic = [y + bar_height / 2 for y in y_positions]
    bars_m = ax.barh(y_mosaic, mosaic_rates, height=bar_height,
                     color="#66bb6a", edgecolor="#2e7d32", linewidth=0.6,
                     label="MOSAIC", zorder=3)

    # ── Value labels ──────────────────────────────────────────────────────
    for bar, val in zip(bars_p, priority_rates):
        x_pos = bar.get_width() + 1
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center",
                fontsize=8, color="#00695c", fontweight="bold")

    for bar, val in zip(bars_m, mosaic_rates):
        x_pos = bar.get_width() + 1
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center",
                fontsize=8, color="#1b5e20", fontweight="bold")

    # ── 100% target line ──────────────────────────────────────────────────
    ax.axvline(100, color="#d32f2f", linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=2, label="100% Target")

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_yticks(y_positions)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Completion Rate (%)", fontsize=11)
    ax.set_xlim(0, 118)
    ax.set_ylim(n_classes - 0.5, -0.5)  # top-to-bottom
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title("Mission-Aware Completion Rate by Workload Class\n(Crisis-Scenario Burst)",
                 fontsize=12, fontweight="bold", pad=12)

    # ── Legend ────────────────────────────────────────────────────────────
    legend = ax.legend(loc="lower right", frameon=True, framealpha=0.9,
                       edgecolor="#bdbdbd", fontsize=9, ncol=1)
    legend.get_frame().set_linewidth(0.5)

    # ── Caption ───────────────────────────────────────────────────────────
    fig.text(0.5, -0.02,
             "Fig. 5. MOSAIC achieves 100% completion for all mission-critical classes\n"
             "while gracefully deferring low-priority workloads. PriorityStrict provides\n"
             "no interference isolation, achieving only 85.7% on inference.",
             ha="center", va="top", fontsize=9, fontstyle="italic", color="#424242")

    # ── Save ──────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = _ROOT / "results" / "fig5_completion_rate.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[viz] Fig. 5 saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fig. 5")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: results/fig5_completion_rate.png)")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    generate_fig5(out)
