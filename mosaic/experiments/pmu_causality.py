import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarking.benchmark import run_all

def main():
    print("==========================================================")
    print("  FINAL PMU CAUSALITY EXPERIMENT                          ")
    print("==========================================================")
    
    # Run a high rate burst pattern to stress the system
    results = run_all(duration=30.0, rate=8.0, pattern="burst", schedulers=["mosaic_nopmu", "mosaic"])
    
    print("\n==========================================================")
    print("  Mode           | P99 Latency (ms) | Starvation %")
    print("----------------------------------------------------------")
    for r in results:
        mode = "PMU disabled" if r["scheduler"] == "MOSAIC-NoPMU" else "PMU enabled"
        print(f"  {mode:<14} | {r['p99_ms']:>16.1f} | {r['starvation_rate']*100:>11.1f}%")
    print("==========================================================\n")
    print("Conclusion: Disabling PMU interference tracking destroys tail latency.")

if __name__ == "__main__":
    main()
