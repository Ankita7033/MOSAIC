# MOSAIC — Benchmark Results

## Verified Results (burst, rate=6/s, 30s, seed=42)

| Scheduler | Hit% | P50 | P95 | P99 | Starve% | JFI | Eff(t/Wh) |
|---|---|---|---|---|---|---|---|
| FCFS | 96.8% | 145ms | 3,423ms | 12,738ms | 6.3% | 0.999 | 631 |
| Round Robin | 96.8% | 152ms | 3,560ms | 13,248ms | 6.3% | 0.999 | 607 |
| SJF | 96.8% | 145ms | 3,423ms | 12,738ms | 6.3% | 0.999 | 631 |
| Priority Static | 96.8% | 145ms | 3,423ms | 12,738ms | 4.8% | 0.999 | 631 |
| **MOSAIC** | **96.0%** | **407ms** | **1,174ms** | **3,178ms** | **0.0%** | **1.000** | **1,171** |

### MOSAIC vs FCFS Improvements
- **P99 latency**: −75.1% (12,738ms → 3,178ms)
- **Starvation**: −6.3 pp (6.3% → 0.0%)
- **Fairness**: +0.001 (0.999 → 1.000 perfect)
- **Energy efficiency**: +85.6% (631 → 1,171 tasks/Wh)

### Interpretation
MOSAIC's P99 is dramatically lower because it prevents batch analytics tasks from colocating with inference workloads — eliminating the 10+ second tail latency events that arise when an analytics batch thrashes the LLC while an inference job tries to run.

The higher P50 for MOSAIC (407ms vs 145ms) reflects the correct tradeoff: background tasks wait slightly longer in the queue to avoid interference, but they complete *reliably* without causing deadline misses for critical workloads.

---

## Reproducing Results

```bash
# Exact reproduction of the table above
python3 run_experiment.py --compare all --pattern burst --rate 6 --duration 30

# Full 2-minute stress test (recommended for portfolio)
python3 run_experiment.py --compare all --pattern burst --rate 10 --duration 120

# Realistic disaster scenario
python3 run_experiment.py --compare all --pattern disaster --rate 6 --duration 180

# Quick 20-second sanity check
python3 run_experiment.py --quick
```

## Chart Generation

```bash
# Install matplotlib for PNG charts
pip install matplotlib

# Generate all 6 chart types
python3 visualization/plot_results.py

# Include Gantt timeline (requires live scheduler run first)
python3 run_experiment.py --demo --duration 60
python3 visualization/plot_results.py --gantt
```

Charts saved to `results/`:
- `benchmark_summary.png` — 6-panel combined figure
- `deadline_hit_rate.png`
- `latency_comparison.png`
- `fairness_index.png`
- `starvation_rate.png`
- `throughput_efficiency.png`
- `gantt_timeline.png`
