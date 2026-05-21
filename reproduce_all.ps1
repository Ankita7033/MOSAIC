# MOSAIC Artifact Reproduction Pipeline

Write-Host "=========================================================="
Write-Host "  MOSAIC Systems Evaluation Artifact Reproducibility Script "
Write-Host "=========================================================="

# 1. Run deterministic baseline benchmark
Write-Host "`n[1/4] Running Deterministic Benchmark (All Baselines)..."
python mosaic\run_experiment.py --compare all --pattern disaster --duration 60 --seed 42

# 2. Run ablation study
Write-Host "`n[2/4] Running MOSAIC Ablation Study..."
python mosaic\run_experiment.py --ablation --pattern burst --duration 30 --seed 42

# 3. Run multi-seed variance analysis
Write-Host "`n[3/4] Running Multi-Seed Evaluation..."
python mosaic\run_experiment.py --seeds 42 43 44 --pattern disaster --duration 20

# 4. Generate all charts and timelines
Write-Host "`n[4/4] Generating Publication-Quality Visualizations..."
python mosaic\visualization\plot_results.py --gantt

Write-Host "`n=========================================================="
Write-Host "  Artifact generation complete!"
Write-Host "  Results and tabular data  -> mosaic\results\"
Write-Host "  Raw traces and PMU logs   -> mosaic\results\raw\"
Write-Host "  Visualizations            -> mosaic\results\*.png"
Write-Host "=========================================================="
