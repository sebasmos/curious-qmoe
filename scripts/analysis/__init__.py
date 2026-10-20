"""
Analysis scripts that read trained runs from scripts/benchmark.py.

- efficiency_benchmark.py: batch-1 latency (median, IQR) and energy per inference of trained checkpoints
- routing_analysis.py: expert routing of held-out test-fold samples
- run_loader.py: rebuilds a trained fold (model as scored, scaled held-out rows) from its run directory
"""
