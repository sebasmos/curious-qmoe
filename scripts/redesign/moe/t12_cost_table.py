"""T12: turn the cost benchmark JSON files into the epoch-time table (markdown). Arithmetic only, no data.
    python t12_cost_table.py COSTDIR > cost_table.md"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = []
for f, bed, n_train, tokens_note in (("cost_urban_L400.json", "UrbanSound8K, 400 frames", 7859, 470), ("cost_L512.json", "ESC-50, 512 frames", 1600, 602), ("cost_L1024.json", "TAU, 1024 frames (device A training rows)", 11520, 1214)):
    for r in json.load(open(d / f)):
        if "k" not in r or "error" in r: continue
        rows.append((bed, r, n_train))
print("| Bed | k | Trainable | Cached tokens | Batch | Step (s) | Clips/s | Epoch (min) | Peak MPS GB (sampled) |\n|---|---|---|---|---|---|---|---|---|")
for bed, r, n in rows:
    ep = n / r["clips_per_s"] / 60
    print(f"| {bed} | {r['k']} | {'LoRA r16' if r['mode']=='lora' else 'full blocks'} ({r['trainable_params']/1e6:.2f} M) | {'yes' if r['cached'] else 'no (full forward each step)'} | {r['bs']} | {r['step_s']:.3f} | {r['clips_per_s']:.0f} | {ep:.2f} | {r['mps_driver_gb_sampled']:.1f} |")
