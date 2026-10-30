"""T14 generic evaluator (ANALYSIS_PLAN_T14.md sections 3, 4 and amendment 1). A series is a DataFrame [cell, seed, fold, f1] with f1 in percent."""
import numpy as np, pandas as pd
import t14_stats as S
def mat(ser, cell):
    g = ser[ser.cell == cell]
    if g.empty: return None
    return g.pivot(index="seed", columns="fold", values="f1").sort_index().sort_index(axis=1).to_numpy(float)
def cellmat(ser, cells):
    """[ncell, nf] seed-averaged F1 per fold."""
    return np.stack([mat(ser, c).mean(0) for c in cells])
def descr(ser, cond):
    X = mat(ser, cond)
    if X is None: return {}
    return dict(mean=float(X.mean()), sd_fold=S.sd_fold(X), sd_seed=S.sd_seed(X), sd_inter=S.sd_inter(X), n_seed=X.shape[0], n_fold=X.shape[1])
def sampling(sys_, ref, cond):
    A, Bm = mat(sys_, cond), mat(ref, cond)
    if A is None or Bm is None or A.shape != Bm.shape or A.shape[1] < 3: return None
    r, ci = S.boot_ratio(A, Bm, "fold"); rs, cis = S.boot_ratio(A, Bm, "seed") if A.shape[0] > 1 else (np.nan, (np.nan, np.nan)); ri, cii = S.boot_ratio(A, Bm, "inter") if A.shape[0] > 1 else (np.nan, (np.nan, np.nan))
    lab = "more stable" if (r < 0.7 and ci[1] < 1) else "less stable" if (r > 1 / 0.7 and ci[0] > 1) else "as stable" if (ci[0] > 0.7 and ci[1] < 1 / 0.7) else "not determinable at five folds"
    return dict(ratio_fold=r, ratio_fold_lo=ci[0], ratio_fold_hi=ci[1], ratio_seed=rs, ratio_seed_lo=cis[0], ratio_seed_hi=cis[1], ratio_inter=ri, ratio_inter_lo=cii[0], ratio_inter_hi=cii[1],
                bf_p=S.bf_p(A, Bm), pm_p=S.pitman_morgan_p(A, Bm), sampling_label=lab)
def shift(sys_, ref, nb_ratio=None, min_cells=3):
    cs = [c for c in sys_.cell.unique() if c in set(ref.cell.unique())]
    if "clean" not in cs or len(cs) < min_cells: return None
    cs = ["clean"] + [c for c in cs if c != "clean"]
    a, b = S.shift_measures(cellmat(sys_, cs), cs), S.shift_measures(cellmat(ref, cs), cs)
    dw = S.paired(a["worst"] - b["worst"], nb_ratio); dd = S.paired(b["drop"] - a["drop"], nb_ratio)
    return dict(n_cells=len(cs), worst_sys=float(a["worst"].mean()), worst_ref=float(b["worst"].mean()), drop_sys=float(a["drop"].mean()), drop_ref=float(b["drop"].mean()),
                sdcells_sys=float(a["sd_cells"].mean()), sdcells_ref=float(b["sd_cells"].mean()), slope_sys=float(np.nanmean(a["slope_per_db"])) if np.isfinite(a["slope_per_db"]).any() else np.nan,
                slope_ref=float(np.nanmean(b["slope_per_db"])) if np.isfinite(b["slope_per_db"]).any() else np.nan,
                dworst=dw["mean"], dworst_lo=dw["lo"], dworst_hi=dw["hi"], dworst_p=dw["p"], dworst_pnb=dw["p_nb"], dworst_folds_ge05=dw["n_ge05"],
                ddrop=dd["mean"], ddrop_lo=dd["lo"], ddrop_hi=dd["hi"], ddrop_p=dd["p"], ddrop_pnb=dd["p_nb"], ddrop_folds_ge05=dd["n_ge05"], n_folds=dw["n"], cells=";".join(cs))
