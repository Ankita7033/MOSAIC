#!/usr/bin/env python3
"""
MOSAIC Distributed Cluster Benchmark
====================================
Simulates a multi-node edge deployment with varying distributed overload scenarios.
Proves that PMU-guided overload isolation scales from single-node to a cluster.
"""
from __future__ import annotations

import time
import threading
import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from cluster.node import EdgeNode
from cluster.coordinator import CentralCoordinator
from benchmarking.benchmark import SCHEDULER_REGISTRY, SimTask
from benchmarking.trace_generator import generate_trace, trace_summary

try:
    import visualization.cluster_dashboard as dash
    dash.start_dashboard_server()
except ImportError:
    dash = None
    print("[Warning] FastAPI/Uvicorn not installed. Live dashboard disabled.")

def execute_distributed_task(task, coordinator: CentralCoordinator, nodes: dict):
    """Worker thread that submits task to coordinator and waits for completion."""
    res = coordinator.dispatch_task(task)
    
    if res == "rejected":
        # Task dropped at admission
        return
        
    # Wait for service
    # In this simulation, we'll let the node's _drain logic handle completion manually
    # to avoid thousands of sleeps, or we can simulate real-time.
    # To keep it consistent with the single node benchmark:
    time.sleep(task.service_ms / 1000.0)
    
    node = nodes[task.assigned_node]
    node.complete_task(task.task_id, task.service_ms)

def run_cluster_experiment(scheduler_name: str, duration: float = 30.0) -> dict:
    print(f"\n[Cluster] Booting 3-Node Edge Cluster with {scheduler_name.upper()} scheduling...")
    
    sched_class = SCHEDULER_REGISTRY.get(scheduler_name.lower())
    if not sched_class:
        raise ValueError(f"Unknown scheduler {scheduler_name}")
        
    nodes = [
        EdgeNode("node_1", sched_class),
        EdgeNode("node_2", sched_class),
        EdgeNode("node_3", sched_class)
    ]
    node_dict = {n.node_id: n for n in nodes}
    
    coordinator = CentralCoordinator(nodes)
    
    # Generate the distributed trace (combining different behaviors to stress different nodes)
    # Node 1 targeted workload
    trace_n1 = generate_trace("cache_thrash", rate=4.0, duration=duration, seed=42)
    # Node 2 targeted workload
    trace_n2 = generate_trace("burst", rate=6.0, duration=duration, seed=43)
    # Node 3 targeted workload
    trace_n3 = generate_trace("poisson", rate=2.0, duration=duration, seed=44)
    
    # Combine and sort by arrival time
    combined_trace = trace_n1 + trace_n2 + trace_n3
    combined_trace.sort(key=lambda t: t["arrival_time"])
    
    start_time = time.monotonic()
    
    threads = []
    
    print(f"[Cluster] Replaying {len(combined_trace)} distributed tasks...")
    for i, td in enumerate(combined_trace):
        now = time.monotonic() - start_time
        if td["arrival_time"] > now:
            time.sleep(td["arrival_time"] - now)
            
        task = SimTask(
            task_id=f"dist_{i+1:06d}",
            cls=td["class"], deadline_ms=td["deadline_ms"],
            service_ms=td["service_ms"], priority=td["priority"],
            tier=td["tier"], arrival_time=td["arrival_time"],
            submit_ms=time.monotonic() * 1000,
        )
            
        th = threading.Thread(target=execute_distributed_task, args=(task, coordinator, node_dict))
        th.start()
        threads.append(th)
        
        # Periodic overload reconciliation (migration check)
        if len(threads) % 10 == 0:
            coordinator.reconcile_overload()
            snapshot = coordinator.collect_telemetry()
            if dash:
                dash.LATEST_TELEMETRY = snapshot
            
    for th in threads:
        th.join()
        
    # Drain remaining
    for n in nodes:
        n.scheduler._drain()
        
    # Aggregate stats
    hits = sum(n.scheduler._hits for n in nodes)
    misses = sum(n.scheduler._misses for n in nodes)
    total = hits + misses
    migrations = sum(n.migrations_out for n in nodes)
    
    print(f"\n[{scheduler_name.upper()}] Cluster Results:")
    print(f"  Total Hit Rate : {(hits/max(1,total))*100:.1f}%")
    print(f"  Total Misses   : {misses}")
    print(f"  Migrations     : {migrations}")
    print(f"  Node 1 Hits    : {nodes[0].scheduler._hits}")
    print(f"  Node 2 Hits    : {nodes[1].scheduler._hits}")
    print(f"  Node 3 Hits    : {nodes[2].scheduler._hits}")
    
    return {
        "scheduler": scheduler_name,
        "hit_rate": hits/max(1,total),
        "misses": misses,
        "migrations": migrations
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=15.0)
    args = p.parse_args()
    
    print("==========================================================")
    print("  MOSAIC Distributed Overload Evaluation                  ")
    print("==========================================================")
    
    results = []
    for sched in ["edf", "priority", "mosaic"]:
        res = run_cluster_experiment(sched, args.duration)
        results.append(res)
        
    print("\n==========================================================")
    print("  FINAL DISTRIBUTED BENCHMARK COMPARISON                  ")
    print("  Scheduler       | Hit Rate | Misses | Migrations        ")
    print("----------------------------------------------------------")
    for r in results:
        print(f"  {r['scheduler']:<15} | {r['hit_rate']*100:>5.1f}%   | {r['misses']:<6} | {r['migrations']}")
    print("==========================================================")

if __name__ == "__main__":
    main()
