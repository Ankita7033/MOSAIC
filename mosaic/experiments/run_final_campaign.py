import sys
import os
import json
import math
import time
import statistics
import threading
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarking.benchmark import run_all, SCHEDULER_REGISTRY
from cluster.node import EdgeNode
from cluster.coordinator import CentralCoordinator
from benchmarking.trace_generator import generate_trace
from benchmarking.cluster_benchmark import run_cluster_experiment
from experiments.overhead_analysis import measure_decision_latency, measure_pmu_overhead, measure_migration_overhead

FINAL_DIR = _ROOT / "results" / "final"

def create_dirs():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

def calc_stats(data):
    if not data: return 0.0, 0.0, 0.0
    mean = statistics.mean(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0
    # 95% CI approx = 1.96 * stdev / sqrt(N)
    ci95 = 1.96 * stdev / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return mean, stdev, ci95

def run_multi_seed(seeds=10):
    print(f"\n--- [1/5] Running Multi-Seed Benchmarks ({seeds} seeds) ---")
    schedulers = ["fcfs", "rr", "sjf", "edf", "priority", "mosaic"]
    
    # Store results: metric -> scheduler -> list of values
    aggregated = {
        "p99_ms": {s: [] for s in schedulers},
        "hit_rate": {s: [] for s in schedulers},
        "starvation": {s: [] for s in schedulers},
        "jfi": {s: [] for s in schedulers}
    }
    
    for seed in range(42, 42 + seeds):
        print(f"  > Seed {seed}...")
        # Using a shorter duration to speed up the campaign for now, 
        # but in reality this would be longer.
        res = run_all(duration=15.0, rate=5.0, pattern="burst", schedulers=schedulers, seed=seed)
        
        for r in res:
            s_name = r["scheduler"].lower().replace("prioritystrict", "priority")
            if s_name not in aggregated["p99_ms"]:
                continue
            aggregated["p99_ms"][s_name].append(r["p99_ms"])
            aggregated["hit_rate"][s_name].append(r["hit_rate"])
            aggregated["starvation"][s_name].append(r["starvation_rate"])
            aggregated["jfi"][s_name].append(r["fairness_index"])

    output = {}
    print("\nMulti-Seed Results (Mean ± 95% CI):")
    for s in schedulers:
        p99_m, _, p99_ci = calc_stats(aggregated["p99_ms"][s])
        hit_m, _, hit_ci = calc_stats(aggregated["hit_rate"][s])
        
        print(f"  {s:<10} | Hit: {hit_m*100:>5.1f}% ±{hit_ci*100:<4.1f}% | P99: {p99_m:>6.1f} ±{p99_ci:<4.1f} ms")
        output[s] = {
            "hit_rate_mean": hit_m, "hit_rate_ci": hit_ci,
            "p99_mean": p99_m, "p99_ci": p99_ci
        }
        
    with open(FINAL_DIR / "multi_seed_stats.json", "w") as f:
        json.dump(output, f, indent=2)

def run_ablations():
    print("\n--- [2/5] Running PMU Causality Ablations ---")
    scheds = ["mosaic", "mosaic_nopmu", "mosaic_noadm"]
    res = run_all(duration=20.0, rate=7.0, pattern="burst", schedulers=scheds, seed=100)
    
    print(f"\n  {'Variant':<15} | {'PMU':<4} | {'Admission':<9} | {'P99 (ms)':>8} | {'Hit%':>6}")
    print("-" * 55)
    
    out_data = []
    for r in res:
        name = r["scheduler"]
        pmu = "OFF" if "NoPMU" in name else "ON"
        adm = "OFF" if "NoAdmission" in name else "ON"
        variant = "Full" if pmu=="ON" and adm=="ON" else name.replace("MOSAIC-", "")
        print(f"  {variant:<15} | {pmu:<4} | {adm:<9} | {r['p99_ms']:>8.1f} | {r['hit_rate']*100:>5.1f}%")
        out_data.append({"variant": variant, "pmu": pmu, "admission": adm, "p99": r["p99_ms"], "hit_rate": r["hit_rate"]})

    with open(FINAL_DIR / "ablations.json", "w") as f:
        json.dump(out_data, f, indent=2)

def run_distributed():
    print("\n--- [3/5] Running Distributed Overload ---")
    res_mosaic = run_cluster_experiment("mosaic", duration=10.0)
    res_priority = run_cluster_experiment("priority", duration=10.0)
    
    out_data = {"mosaic": res_mosaic, "priority": res_priority}
    with open(FINAL_DIR / "distributed.json", "w") as f:
        json.dump(out_data, f, indent=2)

def run_overhead():
    print("\n--- [4/5] Running Overhead Analysis ---")
    med_dec, p99_dec = measure_decision_latency()
    med_pmu = measure_pmu_overhead()
    med_mig = measure_migration_overhead()
    
    print(f"  PMU read            : {med_pmu:.2f} µs")
    print(f"  Scheduling decision : {med_dec:.2f} µs")
    print(f"  Migration           : {med_mig/1000.0:.2f} ms")
    
    out_data = {
        "pmu_read_us": med_pmu,
        "scheduling_decision_us": med_dec,
        "migration_ms": med_mig / 1000.0
    }
    with open(FINAL_DIR / "overhead.json", "w") as f:
        json.dump(out_data, f, indent=2)

def run_real_workloads():
    print("\n--- [5/5] Real Application Workloads Mapping ---")
    print("  Mapping YOLOv8 -> Inference_Critical (High Compute, High Cache)")
    print("  Mapping Redis -> Log_Archive (Low Compute, High Memory)")
    print("  Mapping FastAPI -> Dispatch_API (Variable Compute, Low Latency)")
    
    res = run_all(duration=15.0, rate=5.0, pattern="cache_thrash", schedulers=["mosaic"], seed=99)
    print("\n  Real Workloads (Simulated Footprints) under Cache Thrash:")
    r = res[0]
    for cls, rates in r["class_hit_rates"].items():
        if cls == "inference_critical": print(f"    YOLOv8 (Inference) Hit Rate  : {rates*100:.1f}%")
        if cls == "dispatch_api":       print(f"    FastAPI (Dispatch) Hit Rate  : {rates*100:.1f}%")
        if cls == "log_archive":        print(f"    Redis (Archive) Hit Rate     : {rates*100:.1f}%")
        
    with open(FINAL_DIR / "real_workloads.json", "w") as f:
        json.dump(r["class_hit_rates"], f, indent=2)

def main():
    print("==========================================================")
    print("  MOSAIC FINAL EXPERIMENT CAMPAIGN")
    print("==========================================================")
    create_dirs()
    
    t0 = time.time()
    run_overhead()
    run_ablations()
    run_real_workloads()
    run_distributed()
    run_multi_seed(seeds=5) # Reduced to 5 for time, easily changeable to 30
    
    print("\n==========================================================")
    print(f"  CAMPAIGN COMPLETE in {time.time()-t0:.1f}s")
    print(f"  All results locked in {FINAL_DIR}/")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
