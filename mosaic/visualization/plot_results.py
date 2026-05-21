#!/usr/bin/env python3
"""
MOSAIC Visualization Suite
============================
Auto-generates publication-quality charts from benchmark results.

Charts produced:
  1. Deadline hit rate comparison bar chart
  2. P50/P95/P99 latency grouped bar chart
  3. Jain's Fairness Index comparison
  4. Starvation rate comparison
  5. Throughput & energy efficiency
  6. Per-class hit rate heatmap
  7. Gantt-style task timeline (from metrics.jsonl)
  8. Latency CDF curves

All charts saved to results/ as PNG files.
Also produces a single combined summary figure.

Usage:
    python3 visualization/plot_results.py                     # uses latest benchmark
    python3 visualization/plot_results.py --input results/benchmark_results.json
    python3 visualization/plot_results.py --gantt             # also plot Gantt
"""

from __future__ import annotations

import sys
import json
import math
import argparse
import colorsys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent

# -- Try to import matplotlib ---------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# -- Fallback: pure-text ASCII charts ------------------------------------------

def ascii_bar(label: str, value: float, max_val: float, width: int = 40) -> str:
    filled = int(value / max(0.001, max_val) * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"  {label:<22} {bar} {value:.1f}"


def ascii_comparison(results: list[dict]) -> None:
    print("\n  -- Deadline Hit Rate ----------------------------------------")
    for r in results:
        print(ascii_bar(r["scheduler"], r["hit_rate"]*100, 100))
    print("\n  -- P99 Latency (ms) -----------------------------------------")
    max_p99 = max(r["p99_ms"] for r in results)
    for r in results:
        print(ascii_bar(r["scheduler"], r["p99_ms"], max_p99))
    print("\n  -- Jain's Fairness Index ------------------------------------")
    for r in results:
        print(ascii_bar(r["scheduler"], r["fairness_index"], 1.0))
    print()


# -- Matplotlib charts ---------------------------------------------------------

PALETTE = {
    "MOSAIC":        "#00e5a0",
    "FCFS":          "#ff4d6a",
    "RoundRobin":    "#ffaa00",
    "SJF":           "#a78bfa",
    "PriorityStatic":"#00d4ff",
    "PriorityStrict":"#00d4ff",
    "EDF":           "#ff66cc",
}

SCHEDULER_ORDER = ["FCFS", "RoundRobin", "SJF", "EDF", "PriorityStatic", "PriorityStrict", "MOSAIC"]

def get_color(name: str) -> str:
    return PALETTE.get(name, "#888888")


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161e28",
        "axes.edgecolor":    "#2a3a4a",
        "axes.labelcolor":   "#a0b0c0",
        "xtick.color":       "#7a8a9a",
        "ytick.color":       "#7a8a9a",
        "text.color":        "#e8edf4",
        "grid.color":        "#1e2e3e",
        "grid.linewidth":    0.5,
        "grid.alpha":        1.0,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   "#e8edf4",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":True,
        "figure.dpi":        120,
    })


def sort_results(results: list[dict]) -> list[dict]:
    order = {name: i for i, name in enumerate(SCHEDULER_ORDER)}
    return sorted(results, key=lambda r: order.get(r["scheduler"], 99))


def plot_hit_rate(ax, results: list[dict]):
    results = sort_results(results)
    names   = [r["scheduler"] for r in results]
    values  = [r["hit_rate"] * 100 for r in results]
    colors  = [get_color(n) for n in names]
    bars    = ax.bar(names, values, color=colors, alpha=0.85, width=0.6,
                     edgecolor="none", zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                color="#e8edf4", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Deadline Hit Rate")
    ax.tick_params(axis="x", rotation=15)
    # Highlight MOSAIC
    for i, name in enumerate(names):
        if name == "MOSAIC":
            bars[i].set_edgecolor("#00ff99")
            bars[i].set_linewidth(1.5)


def plot_latency(ax, results: list[dict]):
    results = sort_results(results)
    names   = [r["scheduler"] for r in results]
    n       = len(names)
    x       = list(range(n))
    w       = 0.25

    for i, (key, label, alpha) in enumerate([("p50_ms","P50",0.6),("p95_ms","P95",0.75),("p99_ms","P99",0.9)]):
        vals   = [r[key] for r in results]
        colors = [get_color(nm) for nm in names]
        offset = (i - 1) * w
        bars   = ax.bar([xi + offset for xi in x], vals, w,
                        color=colors, alpha=alpha, edgecolor="none",
                        label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("P50 / P95 / P99 Latency")
    ax.legend(loc="upper left", framealpha=0.3, fontsize=8)


def plot_fairness(ax, results: list[dict]):
    results = sort_results(results)
    names   = [r["scheduler"] for r in results]
    values  = [r["fairness_index"] for r in results]
    colors  = [get_color(n) for n in names]
    bars    = ax.bar(names, values, color=colors, alpha=0.85, width=0.6, edgecolor="none", zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#e8edf4")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Scheduling Fairness (JFI)")
    ax.axhline(1.0, color="#ffffff", linewidth=0.5, linestyle="--", alpha=0.3, label="Perfect fairness")
    ax.tick_params(axis="x", rotation=15)


def plot_starvation(ax, results: list[dict]):
    results = sort_results(results)
    names   = [r["scheduler"] for r in results]
    values  = [r["starvation_rate"] * 100 for r in results]
    colors  = [get_color(n) for n in names]
    bars    = ax.bar(names, values, color=colors, alpha=0.85, width=0.6, edgecolor="none", zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="#e8edf4")
    ax.set_ylabel("Starvation Rate (%)")
    ax.set_title("Task Starvation Rate (lower=better)")
    ax.tick_params(axis="x", rotation=15)


def plot_class_heatmap(ax, results: list[dict]):
    """Per-class hit rate heatmap."""
    from scheduler_classes import CLASS_NAMES as classes
    results    = sort_results(results)
    schedulers = [r["scheduler"] for r in results]
    # Build matrix
    matrix = []
    for r in results:
        row = [r["class_hit_rates"].get(c, 0.0) for c in classes]
        matrix.append(row)
    # Plot
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_","\n") for c in classes], fontsize=7)
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels(schedulers, fontsize=8)
    ax.set_title("Hit Rate by Workload Class")
    # Annotations
    for i in range(len(schedulers)):
        for j in range(len(classes)):
            ax.text(j, i, f"{matrix[i][j]:.0%}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i][j] > 0.5 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_efficiency(ax, results: list[dict]):
    results = sort_results(results)
    names   = [r["scheduler"] for r in results]
    tps     = [r["throughput_tps"] for r in results]
    eff     = [r["efficiency_tpwh"] for r in results]
    x       = list(range(len(names)))
    w       = 0.35
    ax2     = ax.twinx()
    b1 = ax.bar([xi - w/2 for xi in x], tps, w, color=[get_color(n) for n in names],
                alpha=0.7, label="Throughput (TPS)", zorder=3)
    b2 = ax2.bar([xi + w/2 for xi in x], eff, w, color=[get_color(n) for n in names],
                 alpha=0.4, label="Efficiency (tasks/Wh)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set_ylabel("Throughput (TPS)", color="#00e5a0")
    ax2.set_ylabel("Energy Efficiency (tasks/Wh)", color="#ffaa00")
    ax.set_title("Throughput & Energy Efficiency")
    lines = [mpatches.Patch(color="#00e5a0", label="Throughput"),
             mpatches.Patch(color="#ffaa00", alpha=0.6, label="Efficiency")]
    ax.legend(handles=lines, fontsize=7, loc="upper left", framealpha=0.3)


def generate_all_charts(results: list[dict], out_dir: Path) -> list[Path]:
    if not HAS_MPL:
        print("[viz] matplotlib not available -- using ASCII charts")
        ascii_comparison(results)
        return []

    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # -- Combined summary figure (6 panels) -----------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_hit_rate(ax1, results)
    plot_latency(ax2, results)
    plot_fairness(ax3, results)
    plot_starvation(ax4, results)
    plot_efficiency(ax5, results)

    # Panel 6: improvement summary text
    mosaic = next((r for r in results if r["scheduler"]=="MOSAIC"), None)
    best   = max((r for r in results if r["scheduler"]!="MOSAIC"),
                 key=lambda r: r["hit_rate"], default=None)
    if mosaic and best:
        ax6.set_facecolor("#0d1117")
        ax6.axis("off")
        lines = [
            ("MOSAIC vs Best Baseline", "title"),
            ("", ""),
            (f"Hit rate: {mosaic['hit_rate']:.1%} vs {best['hit_rate']:.1%}", "metric"),
            (f"Improvement: +{(mosaic['hit_rate']-best['hit_rate'])*100:.1f} pp", "good"),
            ("", ""),
            (f"P99 latency: {mosaic['p99_ms']:.0f}ms vs {best['p99_ms']:.0f}ms", "metric"),
            (f"Reduction: {(best['p99_ms']-mosaic['p99_ms'])/max(1,best['p99_ms'])*100:.1f}%", "good"),
            ("", ""),
            (f"Starvation: {mosaic['starvation_rate']:.1%} vs {best['starvation_rate']:.1%}", "metric"),
            (f"Jain's FI: {mosaic['fairness_index']:.3f} vs {best['fairness_index']:.3f}", "metric"),
        ]
        y = 0.95
        for text, kind in lines:
            color = {"title":"#00e5a0","metric":"#e8edf4","good":"#00ff88","":"#555"}.get(kind,"#aaa")
            size  = 13 if kind == "title" else 10
            weight = "bold" if kind in ("title","good") else "normal"
            ax6.text(0.05, y, text, transform=ax6.transAxes,
                     color=color, fontsize=size, fontweight=weight, va="top")
            y -= 0.09

    fig.suptitle("MOSAIC -- Scheduler Benchmark Results (Disaster-Response Edge)",
                 fontsize=14, fontweight="bold", color="#e8edf4", y=0.98)

    # Legend strip
    legend_handles = [mpatches.Patch(color=get_color(n), label=n)
                      for n in SCHEDULER_ORDER if any(r["scheduler"]==n for r in results)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               framealpha=0.2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    path = out_dir / "benchmark_summary.png"
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    generated.append(path)
    print(f"[viz] Saved: {path}")

    # -- Individual charts -----------------------------------------------------
    individual = [
        ("hit_rate",    plot_hit_rate,    "deadline_hit_rate.png"),
        ("latency",     plot_latency,     "latency_comparison.png"),
        ("fairness",    plot_fairness,    "fairness_index.png"),
        ("starvation",  plot_starvation,  "starvation_rate.png"),
        ("efficiency",  plot_efficiency,  "throughput_efficiency.png"),
    ]
    for _, fn, fname in individual:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#0d1117")
        fn(ax, results)
        p = out_dir / fname
        fig.savefig(p, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        generated.append(p)
        print(f"[viz] Saved: {p}")

    return generated


def plot_gantt(events: list[dict], out_dir: Path, max_tasks: int = 40) -> Optional[Path]:
    """
    Gantt-style timeline from timeline events.
    Each row = one task, coloured by workload class.
    Shows critical window, grouped by class.
    """
    if not HAS_MPL or not events:
        return None

    # Build task intervals
    admit_ts:    dict[str, float] = {}
    complete_ts: dict[str, float] = {}
    reject_ts:   dict[str, float] = {}
    expire_ts:   dict[str, float] = {}
    task_class:  dict[str, str]   = {}

    for e in events:
        tid = e.get("task_id")
        if not tid: continue
        task_class[tid] = e.get("class","unknown")
        
        if e["event"] == "ADMIT":
            admit_ts[tid] = e["ts"]
        elif e["event"] == "COMPLETE" and tid in admit_ts:
            complete_ts[tid] = e["ts"]
        elif e["event"] == "REJECT":
            reject_ts[tid] = e["ts"]
        elif e["event"] == "EXPIRE":
            expire_ts[tid] = e["ts"]

    tasks = [(tid, admit_ts[tid], complete_ts.get(tid, admit_ts[tid]), task_class[tid])
             for tid in admit_ts]
    tasks.sort(key=lambda x: (x[3], x[1])) # Group by class, then start time
    tasks = tasks[:max_tasks]

    if not tasks:
        return None

    cls_colors = {
        "inference_critical": "#ff4d6a",
        "dispatch_api":       "#00e5a0",
        "sensor_fusion":      "#00d4ff",
        "analytics_batch":    "#ffaa00",
        "model_update":       "#a78bfa",
        "log_archive":        "#7a8a9a",
    }

    min_ts = min(t[1] for t in tasks)
    setup_style()
    fig, ax = plt.subplots(figsize=(14, max(4, len(tasks) * 0.22)))
    fig.patch.set_facecolor("#0d1117")

    for i, (tid, start, end, cls) in enumerate(tasks):
        s = start / 1000.0
        d = max(0.01, (end - start) / 1000.0)
        color = cls_colors.get(cls, "#888888")
        ax.barh(i, d, left=s, height=0.7, color=color, alpha=0.85, edgecolor="none")
        ax.text(s + d + 0.05, i, tid, va="center", fontsize=6, color="#a0a0a0")

        # Rejection & Expiration Markers
        if tid in reject_ts:
            ax.plot(reject_ts[tid]/1000.0, i, "rx", markersize=8, markeredgewidth=2)
        if tid in expire_ts:
            ax.plot(expire_ts[tid]/1000.0, i, "yo", markersize=6)

    ax.set_xlabel("Time (seconds)")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[3].replace("_","\n") for t in tasks], fontsize=6)
    ax.set_title("MOSAIC Task Execution Timeline (Gantt)")
    ax.invert_yaxis()

    legend_handles = [mpatches.Patch(color=c, label=cls.replace("_"," "))
                      for cls, c in cls_colors.items()]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.3, ncol=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gantt_timeline.png"
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved: {path}")
    return path

def plot_timeline_metrics(events: list[dict], out_dir: Path) -> Optional[Path]:
    """Plots active tasks and queue depths over time."""
    if not HAS_MPL or not events:
        return None

    events.sort(key=lambda e: e["ts"])
    times = []
    active = []
    
    current_active = 0
    for e in events:
        t_sec = e["ts"] / 1000.0
        times.append(t_sec)
        
        if e["event"] == "ADMIT": current_active += 1
        elif e["event"] in ("COMPLETE", "EXPIRE"): current_active = max(0, current_active - 1)
        
        active.append(current_active)

    if not times: return None

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d1117")

    ax.plot(times, active, color="#00e5a0", linewidth=1.5, label="Active Tasks")
    ax.fill_between(times, 0, active, color="#00e5a0", alpha=0.2)
    
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Running Tasks")
    ax.set_title("MOSAIC Active Task Timeline")
    ax.legend(loc="upper right", framealpha=0.3)

    path = out_dir / "time_series_metrics.png"
    fig.savefig(path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved: {path}")
    return path


# Stub for class names when imported standalone
try:
    import scheduler_classes
except ImportError:
    class _stub:
        CLASS_NAMES = ["inference_critical","dispatch_api","sensor_fusion",
                       "analytics_batch","model_update","log_archive"]
    sys.modules["scheduler_classes"] = _stub()  # type: ignore


def main():
    p = argparse.ArgumentParser(description="MOSAIC Visualization Suite")
    p.add_argument("--input",   default=str(_ROOT/"results"/"benchmark_results.json"))
    p.add_argument("--out-dir", default=str(_ROOT/"results"))
    p.add_argument("--gantt",   action="store_true")
    args = p.parse_args()

    results_path = Path(args.input)
    out_dir      = Path(args.out_dir)

    if not results_path.exists():
        print(f"[viz] No results found at {results_path}")
        print(f"      Run: python3 benchmarking/benchmark.py first")
        sys.exit(1)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    results = data.get("results", data) if isinstance(data, dict) else data

    print(f"\n[viz] Generating charts from {results_path}")
    generated = generate_all_charts(results, out_dir)

    if args.gantt and "results" in data:
        # Extract MOSAIC timeline
        mosaic_res = next((r for r in results if r.get("scheduler") == "MOSAIC"), None)
        if mosaic_res and "timeline" in mosaic_res:
            plot_gantt(mosaic_res["timeline"], out_dir)
            plot_timeline_metrics(mosaic_res["timeline"], out_dir)

    if generated:
        print(f"\n[viz] {len(generated)} charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
