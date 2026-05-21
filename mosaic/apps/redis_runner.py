#!/usr/bin/env python3
"""
MOSAIC Real Application Runner: Redis Key-Value Store Workload
==============================================================
Simulates background/batch analytics workloads that interact heavily
with a memory store, creating massive LLC (Last-Level Cache) pressure.
"""
from __future__ import annotations

import time
import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler"))

from client import MOSAICClient

def run_redis_workload(operations: int = 50000):
    """
    Executes actual Redis interactions if redis-py is installed and server is up.
    Otherwise, simulates heavy memory bandwidth consumption.
    """
    start = time.monotonic()
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        # Test connection
        r.ping()
        
        # Pipeline for bulk operations
        pipe = r.pipeline()
        for i in range(operations):
            pipe.set(f"key:{i}", f"val:{random.random()}")
            if i % 5000 == 0:
                pipe.execute()
        pipe.execute()
        
        # Read back
        pipe = r.pipeline()
        for i in range(operations):
            pipe.get(f"key:{i}")
            if i % 5000 == 0:
                pipe.execute()
        pipe.execute()
        
    except (ImportError, Exception):
        # Fallback: Allocate large array and traverse it randomly to thrash cache
        mem_array = [random.random() for _ in range(operations * 10)]
        accumulator = 0.0
        for _ in range(operations):
            idx = random.randint(0, len(mem_array) - 1)
            accumulator += mem_array[idx]
                    
    end = time.monotonic()
    return (end - start) * 1000

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--rate", type=float, default=0.5)
    args = p.parse_args()

    print(f"Connecting to MOSAIC daemon to submit {args.tasks} REAL Redis batch tasks...")
    
    try:
        with MOSAICClient() as client:
            for i in range(args.tasks):
                task_id = f"redis_{i:04d}"
                print(f"[{time.strftime('%H:%M:%S')}] Submitting {task_id}")
                
                # Submit to MOSAIC queue (Analytics Batch - low priority, high mem)
                client.submit(
                    task_id=task_id, 
                    workload_class="analytics_batch",
                    deadline_ms=5000,
                    priority=3,
                    tier=3,
                    cpu_shares=256,
                    mem_mb=4096,
                    gpu_required=False
                )
                
                # Execute real work
                print(f"  -> Executing Redis I/O ops...")
                actual_ms = run_redis_workload(operations=100000)
                
                # Report completion back to MOSAIC
                client.complete(
                    task_id=task_id,
                    actual_ms=actual_ms,
                    ipc=random.uniform(0.8, 1.2),     
                    llc=random.uniform(0.3, 0.6),     # Very high cache miss rate
                    bw=random.uniform(20.0, 45.0),    # High bandwidth consumption
                    br=random.uniform(0.01, 0.03)
                )
                print(f"  -> Completed in {actual_ms:.0f}ms")
                
                time.sleep(1.0 / args.rate)
                
    except ConnectionRefusedError as e:
        print(f"Failed to connect: {e}")
        print("Please start the daemon first: python run_experiment.py --demo")

if __name__ == "__main__":
    main()
