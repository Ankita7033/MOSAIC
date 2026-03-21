#!/usr/bin/env python3
"""
MOSAIC -- Main Experiment Runner
================================
Single entry point for the entire MOSAIC pipeline.

import sys as _sys
if _sys.platform == "win32":
    import re as _re, builtins as _bi
    _op = _bi.print
    def _wp(*a, **k): _op(*[_re.sub("\033[^m]*m","",str(x)) for x in a], **k)
    _bi.print = _wp

Usage:
    python3 run_experiment.py --compare all
    python3 run_experiment.py --compare all --pattern burst --rate 8 --duration 90
    python3 run_experiment.py --quick
    python3 run_experiment.py --classify --ipc 1.8 --llc 0.42 --bw 28.0 --br 0.08
    python3 run_experiment.py --demo      # 30s quick demo

This script:
  1. Runs all 5 schedulers with identical workloads
  2. Computes all metrics (hit rate, latency, JFI, starvation, efficiency)
  3. Auto-generates comparison charts (PNG)
  4. Prints the full comparison table
  5. Optionally starts the live scheduler daemon + workload generator
"""

from __future__ import annotations

import sys
import os
import json
import time
import math
import random
import argparse
import subprocess
import threading
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
sys.path.insert(0, str(_ROOT / "scheduler"))
sys.path.insert(0, str(_ROOT / "benchmarking"))
sys.path.insert(0, str(_ROOT / "visualization"))
sys.path.insert(0, str(_ROOT / "workload-gen"))

from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES

# -- ANSI Colors ----------------------------------------------------------------
G  = ""; C  = ""; Y  = ""
R  = ""; B  = ""; M  = ""
W  = ""; DIM= "";  RST= ""; BOLD=""

def banner():
    print(f"""
{C}{BOLD}
  ███╗   ███╗ ██████╗ ███████╗ █████╗ ██╗ ██████╗
  ████╗ ████║██╔===██╗██╔====╝██╔==██╗██║██╔====╝
  ██╔████╔██║██║   ██║███████╗███████║██║██║
  ██║╚██╔╝██║██║   ██║╚====██║██╔==██║██║██║
  ██║ ╚=╝ ██║╚██████╔╝███████║██║  ██║██║╚██████╗
  ╚=╝     ╚=╝ ╚=====╝ ╚======╝╚=╝  ╚=╝╚=╝ ╚=====╝
{RST}
  {W}Multi-Objective Scheduler for AI-Cloud Inference Colocation{RST}
  {DIM}Disaster-Response Edge Computing Domain{RST}
""")

def log_step(msg): print(f"\n{G}{BOLD}[MOSAIC]{RST} {msg}")
def log_info(msg): print(f"         {C}{msg}{RST}")
def log_warn(msg): print(f"         {Y}[!]  {msg}{RST}")
def log_ok(msg):   print(f"         {G}[OK]  {msg}{RST}")
def log_err(msg):  print(f"         {R}[X]  {msg}{RST}")


# -- ML Classifier demo ---------------------------------------------------------

def run_classify(ipc, llc, bw, br):
    from ml_classifier import WorkloadClassifier
    clf = WorkloadClassifier()
    r   = clf.classify(ipc, llc, bw, br)
    print(f"\n  {W}ML Workload Classification Result{RST}")
    print(f"  {'-'*40}")
    print(f"  Input  : IPC={ipc}  LLC_miss={llc}  MemBW={bw}GB/s  branch_miss={br}")
    print(f"  Class  : {G}{r.predicted_class}{RST}")
    print(f"  Tier   : {r.tier}  ({'CRITICAL' if r.tier==1 else 'URGENT' if r.tier==2 else 'IMPORTANT' if r.tier==3 else 'BACKGROUND'})")
    print(f"  Conf   : {r.confidence:.1%}")
    print(f"  Method : {r.method}")
    print(f"\n  All distances:")
    for cls, dist in sorted(r.distances.items(), key=lambda x: x[1]):
        bar = "█" * int((1 - dist) * 20) if dist < 1 else ""
        print(f"    {cls:<25} {dist:.4f}  {G if cls==r.predicted_class else DIM}{bar}{RST}")


# -- Full benchmark pipeline ----------------------------------------------------

def run_compare_all(pattern: str, rate: float, duration: float,
                    quick: bool, no_charts: bool) -> dict:
    from benchmark import run_all, print_comparison_table, SCHEDULER_REGISTRY

    if quick:
        duration = 20.0
        rate     = 4.0

    log_step(f"Running all 5 schedulers  pattern={pattern}  rate={rate}/s  dur={duration}s")
    log_info("Schedulers: FCFS | Round Robin | SJF | Priority | MOSAIC")
    log_info("Metrics: hit_rate | p99 | JFI | starvation | throughput | energy")
    print()

    t0      = time.monotonic()
    results = run_all(duration, rate, pattern,
                      ["fcfs","rr","sjf","priority","mosaic"])
    elapsed = time.monotonic() - t0

    print_comparison_table(results)

    # Save
    out = {
        "config": {"pattern": pattern, "rate": rate, "duration": duration},
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results_path = _ROOT / "results" / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2))
    log_ok(f"Results saved → {results_path}")

    # Charts
    if not no_charts:
        log_step("Generating charts...")
        try:
            from plot_results import generate_all_charts, plot_gantt
            generated = generate_all_charts(results, _ROOT / "results")
            gantt_path = _ROOT / "data" / "metrics.jsonl"
            if gantt_path.exists():
                plot_gantt(gantt_path, _ROOT / "results")
            if generated:
                log_ok(f"{len(generated)} charts saved to results/")
            else:
                log_warn("Install matplotlib for PNG charts: pip install matplotlib")
        except Exception as e:
            log_warn(f"Chart generation failed: {e}")

    return out


# -- Live scheduler demo --------------------------------------------------------

def run_live_demo(pattern: str, rate: float, duration: float) -> None:
    """Start daemon + workload generator, run live for <duration> seconds."""
    log_step("Starting MOSAIC scheduler daemon...")

    # Kill any existing daemon
    sock_path = _ROOT / "data" / "mosaic.sock"
    if sock_path.exists():
        sock_path.unlink()

    daemon_log = _ROOT / "data" / "daemon.log"
    (_ROOT / "data").mkdir(exist_ok=True)
    daemon = subprocess.Popen(
        [sys.executable, str(_ROOT / "scheduler" / "scheduler.py")],
        stdout=open(daemon_log, "w"), stderr=subprocess.STDOUT,
    )

    # Wait for socket
    for _ in range(30):
        if sock_path.exists():
            break
        time.sleep(0.3)

    if not sock_path.exists():
        log_err(f"Daemon failed to start. See {daemon_log}")
        return

    log_ok(f"Daemon PID={daemon.pid}")
    log_step(f"Running workload generator  pattern={pattern}  rate={rate}/s  dur={duration}s")

    try:
        subprocess.run([
            sys.executable, str(_ROOT / "workload-gen" / "workload_gen.py"),
            "--pattern", pattern,
            "--rate", str(rate),
            "--duration", str(duration),
            "--scheduler",
            "--output", str(_ROOT / "results" / "live_run.json"),
        ])
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)
        log_ok("Daemon stopped")


# -- Quick classifier test ------------------------------------------------------

def run_classifier_tests() -> None:
    from ml_classifier import WorkloadClassifier
    clf = WorkloadClassifier()

    test_cases = [
        (1.8, 0.42, 28.0, 0.08, "inference_critical", "Drone image inference"),
        (2.9, 0.06,  3.2, 0.04, "dispatch_api",       "Responder API call"),
        (2.1, 0.18,  8.5, 0.06, "sensor_fusion",      "GPS+thermal fusion"),
        (1.4, 0.28, 12.0, 0.05, "analytics_batch",    "Damage heatmap"),
        (1.6, 0.38, 22.0, 0.07, "model_update",       "Edge model finetune"),
        (1.1, 0.08,  4.0, 0.03, "log_archive",        "Incident log sync"),
    ]

    print(f"\n  {W}ML Workload Classifier -- Accuracy Test{RST}")
    print(f"  {'-'*65}")
    correct = 0
    for ipc, llc, bw, br, expected, desc in test_cases:
        r  = clf.classify(ipc, llc, bw, br)
        ok = r.predicted_class == expected
        if ok: correct += 1
        sym = f"{G}[OK]{RST}" if ok else f"{R}[X]{RST}"
        print(f"  {sym} {desc:<28} → {r.predicted_class:<22} {r.confidence:.0%} conf")

    pct = correct / len(test_cases)
    col = G if pct >= 0.9 else Y if pct >= 0.7 else R
    print(f"\n  {col}Accuracy: {correct}/{len(test_cases)} = {pct:.0%}{RST}")


# -- Entry Point ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MOSAIC -- Disaster-Response Edge Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_experiment.py --compare all
  python3 run_experiment.py --compare all --pattern burst --rate 10 --duration 120
  python3 run_experiment.py --quick
  python3 run_experiment.py --demo
  python3 run_experiment.py --classify --ipc 1.8 --llc 0.42 --bw 28.0 --br 0.08
  python3 run_experiment.py --test-classifier
        """,
    )
    parser.add_argument("--compare",    nargs="+", metavar="SCHED",
                        help="Run comparison: 'all' or subset of fcfs rr sjf priority mosaic")
    parser.add_argument("--pattern",    default="burst",
                        choices=["poisson","burst","sinusoidal","disaster","step"])
    parser.add_argument("--rate",       type=float, default=6.0)
    parser.add_argument("--duration",   type=float, default=60.0)
    parser.add_argument("--quick",      action="store_true",
                        help="Short run: 20s, rate=4")
    parser.add_argument("--demo",       action="store_true",
                        help="Live daemon + workload generator demo")
    parser.add_argument("--no-charts",  action="store_true")
    parser.add_argument("--classify",   action="store_true",
                        help="Classify a single workload from perf counters")
    parser.add_argument("--ipc",        type=float, default=2.1)
    parser.add_argument("--llc",        type=float, default=0.18)
    parser.add_argument("--bw",         type=float, default=8.5)
    parser.add_argument("--br",         type=float, default=0.06)
    parser.add_argument("--test-classifier", action="store_true")
    args = parser.parse_args()

    banner()

    if args.classify:
        run_classify(args.ipc, args.llc, args.bw, args.br)

    elif args.test_classifier:
        run_classifier_tests()

    elif args.demo:
        dur = 30.0 if args.quick else args.duration
        run_live_demo(args.pattern, args.rate, dur)

    elif args.compare:
        run_compare_all(args.pattern, args.rate, args.duration,
                        args.quick, args.no_charts)

    else:
        # Default: run everything
        log_info("No option specified -- running full benchmark (use --compare all)")
        run_compare_all(args.pattern, args.rate, args.duration,
                        args.quick, args.no_charts)

    log_step("Done.")


if __name__ == "__main__":
    main()
