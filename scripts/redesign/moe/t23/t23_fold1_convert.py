"""T23 amendment 1: the fold-1 tails of the P1 re-check were saved as FP32 state dicts (tail.pt). Convert each to the compact W8A8 form the
other folds use (tail_q8.pt) and write its training rows (train_rows.npy, recomputed by the fine-tune script's rule). The equality of
TailQ8(tail_q8.pt) on the W8A8 trunk cache with the P1 file wg__clean.npy is checked on the GPU by t23_ft_score.py.   python -I t23_fold1_convert.py R"""
import copy, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
for p_ in (ROOT, HERE.parent, HERE): sys.path.insert(0, str(p_))
import numpy as np, pandas as pd, torch  # noqa: E402
from QWave.panns_wavegram import load_wavegram  # noqa: E402
from t23_tailq import save_tail_q8  # noqa: E402
from t13_common import sdir, tau_file  # noqa: E402
R = Path(sys.argv[1]); DS = Path.home() / "Documents/DATASETS"
base = load_wavegram(Path.home() / "Documents/qmoe-weights/panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth")
META = {"esc50": DS / "ESC-50/ast_768/esc-50.csv", "urban": DS / "Urban8k/ast_768/urbansound8k.csv", "tau": DS / "TAU2020Mobile/ast_768/tau2020mobile.csv"}
for task in ("esc50", "urban", "tau"):
    meta = pd.read_csv(META[task]); rows = np.flatnonzero((meta.device == "a").to_numpy()) if task == "tau" else np.arange(len(meta))
    for seed in (0, 1, 2):
        od = R / f"ft/{task}/f1_s{seed}"
        if (od / "tail_q8.pt").exists(): continue
        sd = torch.load(od / "tail.pt", map_location="cpu", weights_only=True)
        cb, fc = copy.deepcopy(base.conv_block6), copy.deepcopy(base.fc1)
        cb.load_state_dict({k[12:]: v.cpu() for k, v in sd.items() if k.startswith("conv_block6.")}, strict=True)
        fc.load_state_dict({k[4:]: v.cpu() for k, v in sd.items() if k.startswith("fc1.")}, strict=True)
        save_tail_q8(cb, fc, od / "tail_q8.pt")
        ref = np.load(tau_file("mn10", seed)) if task == "tau" else np.load(sdir(task, "mn10", seed)[0] / "mn10_w8a8__clean.npz")
        iv = ref["iv_idx_1"].astype(int); te = np.flatnonzero(meta.folder.to_numpy()[rows] == 1)
        np.save(od / "train_rows.npy", rows[np.setdiff1d(np.setdiff1d(np.arange(len(rows)), te), iv)])
        print(task, seed, "converted", flush=True)
