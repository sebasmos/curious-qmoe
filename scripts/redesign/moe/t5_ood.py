"""T5 step 3b (ANALYSIS_PLAN_T3T5.md section 7): unseen-class detection (E7 sets) for the cheap margin, the pair's mixture margin, noise_lr and random. Per (task, seed, fold): AUROC of "clip is from the
other dataset's unseen classes" against the in-domain test fold. Heads are the stored `esc` heads (a forward pass of stored embeddings through stored heads, nothing trained).
Reproduction check: with the final round's own head kind this must equal its stored `e7_pairs.csv`.   python t5_ood.py --out DIR [--check-only]"""
import argparse, json, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_common import *
import torch
from sklearn.metrics import roc_auc_score
import realshift as RS            # Stage 2's Heads, class maps
torch.set_num_threads(2)
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--check-only", action="store_true"); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
I5 = Q / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"; S3C_ = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"; S3B_ = S3B
SHARED, OTHER, AST_CSV, STATS_DS = RS.SHARED, RS.OTHER, RS.AST_CSV, RS.STATS_DS
def stems_of(ds): return [Path(s).stem for s in pd.read_csv(AST_CSV[ds], usecols=["name"])["name"]]
def load_emb(path, names):
    X = np.load(path); side = json.loads(path.with_suffix(".json").read_text())
    if side["names_crc32"] != zlib.crc32("\n".join(names).encode()) or len(X) != len(names) or not np.isfinite(X).all(): raise SystemExit(f"{path}: rows do not match the dataset CSV")
    return X
def margin(p): s = np.sort(p, 1); return s[:, -1] - s[:, -2]
def cheap_emb(ds): return (S3B_ / "emb_cnn/esc50/mn10_w8a8__clean.npy") if ds == "esc50" else (S3C_ / "emb_cnn/urban8k_unpadded/mn10_w8a8__clean.npy")
def cheap_run(ds, s): return (S3B_ / "outputs" / f"mn10_esc50_s{s}") if ds == "esc50" else (S3C_ / "outputs_unp" / f"mn10unp_urban_s{s}")
def ast_run(ds, s): return I5 / "outputs" / (f"astesc50_s{s}" if ds == "esc50" else f"asturban_s{s}")
SEEDS = {"esc50": [0, 1, 2, 3, 4], "urban8k": [0, 1, 2]}
rows = []
for ds in ("esc50", "urban8k"):
    other = OTHER[ds]; names_in, names_o = stems_of(ds), stems_of(other)
    meta_o = pd.read_csv(AST_CSV[other], usecols=["label"]); ood = np.flatnonzero(~meta_o["label"].isin(SHARED[other]).to_numpy())
    Xc_in, Xc_o = load_emb(cheap_emb(ds), names_in), load_emb(cheap_emb(other), names_o)[ood]
    Xa_in = load_emb(I5 / "emb" / ds / "w4g128__clean.npy", names_in); Xa_o = load_emb(I5 / "emb" / other / "w4g128__clean.npy", names_o)[ood]
    a_in = A.AudioStats(I7 / "audio_stats" / STATS_DS[ds], names_in); st_o = np.load(I7 / "audio_stats" / STATS_DS[other] / "clean.npz", allow_pickle=True)
    if [str(s) for s in st_o["names"]] != names_o: raise SystemExit("audio statistics rows differ")
    S_o = st_o["stats"][ood]; avv = A.field_arrays(len(names_in))[1]
    for seed in SEEDS[ds]:
        hc = RS.Heads(cheap_run(ds, seed) / "esc"); ha = RS.Heads(ast_run(ds, seed) / "esc")
        for f, (te, iv, _, _) in hc.folds.items():
            assert np.array_equal(te, ha.folds[f][0]) and np.array_equal(iv, ha.folds[f][1]), (ds, seed, f, "head splits differ")
            pc_te, pc_o = hc.probs(f, Xc_in[te]), hc.probs(f, Xc_o); pa_te, pa_o = ha.probs(f, Xa_in[te]), ha.probs(f, Xa_o)
            pm_te, pm_o = 0.5 * (pc_te + pa_te), 0.5 * (pc_o + pa_o)
            noise = A.fit_lr(a_in.rows("field", iv, "val"), (avv[iv] != "clean").astype(int)); s_te, s_o = noise(a_in.rows("clean", te, "test")), noise(S_o)
            lab = np.r_[np.zeros(len(te)), np.ones(len(ood))]
            rows.append(dict(task="esc50" if ds == "esc50" else "urban", seed=seed, fold=f, n_in=len(te), n_ood=len(ood), auroc_cheap_margin=roc_auc_score(lab, np.r_[1 - margin(pc_te), 1 - margin(pc_o)]),
                             auroc_cheap_msp=roc_auc_score(lab, np.r_[1 - pc_te.max(1), 1 - pc_o.max(1)]), auroc_mix_margin=roc_auc_score(lab, np.r_[1 - margin(pm_te), 1 - margin(pm_o)]),
                             auroc_mix_msp=roc_auc_score(lab, np.r_[1 - pm_te.max(1), 1 - pm_o.max(1)]), auroc_ast_margin=roc_auc_score(lab, np.r_[1 - margin(pa_te), 1 - margin(pa_o)]),
                             auroc_noise_lr=roc_auc_score(lab, np.r_[s_te, s_o]), auroc_random=0.5))
        print(ds, "seed", seed, flush=True)
D = pd.DataFrame(rows); D.to_csv(a.out / "t5_ood_fold.csv", index=False)
# reproduction check against the final round: UrbanSound8K task uses the same cheap head (`esc`) and noise_lr; compare margin and noise_lr AUROC with e7_pairs.csv (family A5, urban)
E7 = pd.read_csv(Q / "2026-10-24_1507_redesign-idea6-final_code-ed82400/analysis/e7_pairs.csv"); E7 = E7[(E7.task == "urban")]
m = D[D.task == "urban"].merge(E7[["seed", "fold", "auroc_margin", "auroc_noise_lr"]], on=["seed", "fold"], suffixes=("", "_stored"))
chk = dict(n=len(m), max_abs_diff_margin=float((m.auroc_cheap_margin - m.auroc_margin).abs().max()), max_abs_diff_noise_lr=float((m.auroc_noise_lr - m.auroc_noise_lr_stored).abs().max()))
json.dump(chk, open(a.out / "t5_ood_repro_check.json", "w"), indent=1); print(chk)
print(D.groupby("task")[[c for c in D.columns if c.startswith("auroc")]].mean().round(4).to_string())
