#!/usr/bin/env python3
"""
MOSAIC Real Application Runner: Flask/FastAPI Service
=====================================================
Simulates a high-frequency, low-latency API dispatch workload.
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

def run_api_workload():
    """
    Simulates parsing JSON payloads, handling web requests, and routing.
    This is highly IPC bound, low memory footprint.
    """
    start = time.monotonic()
    
    import json
    
    # Simulate processing 100 small HTTP JSON requests
    for _ in range(100):
        dummy_payload = '{"user_id": 12345, "action": "update", "data": {"key": "value", "items": [1, 2, 3, 4, 5]}}'
        parsed = json.loads(dummy_payload)
        parsed["timestamp"] = time.time()
        _ = json.dumps(parsed)
        
    end = time.monotonic()
    return (end - start) * 1000

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=100)
    p.add_argument("--rate", type=float, default=20.0) # 20 req/s
    args = p.parse_args()

    print(f"Connecting to MOSAIC daemon to submit {args.tasks} REAL API requests...")
    
    try:
        with MOSAICClient() as client:
            for i in range(args.tasks):
                task_id = f"api_{i:04d}"
                
                # Submit to MOSAIC queue (Dispatch API - very short deadline)
                client.submit(
                    task_id=task_id, 
                    workload_class="dispatch_api",
                    deadline_ms=50,
                    priority=2,
                    tier=2,
                    cpu_shares=512,
                    mem_mb=128,
                    gpu_required=False
                )
                
                actual_ms = run_api_workload()
                
                # Report completion back to MOSAIC
                client.complete(
                    task_id=task_id,
                    actual_ms=actual_ms,
                    ipc=random.uniform(2.5, 3.5),     # High IPC
                    llc=random.uniform(0.01, 0.05),   # Low cache miss
                    bw=random.uniform(1.0, 3.0),      # Low bandwidth
                    br=random.uniform(0.01, 0.05)
                )
                
                # Print every 10th request to avoid console spam
                if i % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Processed {task_id} in {actual_ms:.1f}ms")
                
                time.sleep(1.0 / args.rate)
                
    except ConnectionRefusedError as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    main()
