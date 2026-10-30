"""Check (a): compare ORT FP32, ORT int8 and torch fake-quant W8A8 embeddings with the torch FP32 embeddings."""
import json, sys
import numpy as np
S3B = sys.argv[1]; R = sys.argv[2]; out = sys.argv[3]
res = {}
for ds, models in (("esc50", ["mn04", "mn10", "mn20"]), ("urban8k", ["mn10"])):
    for m in models:
        import os
        E = S3B if os.path.exists(f"{S3B}/emb_cnn/{ds}/{m}_fp32__clean.npy") else R
        fp = np.load(f"{E}/emb_cnn/{ds}/{m}_fp32__clean.npy").astype(np.float64)
        w8 = np.load(f"{E}/emb_cnn/{ds}/{m}_w8a8__clean.npy").astype(np.float64)
        o32 = np.load(f"{R}/emb_int8/{ds}/{m}_ortfp32__clean.npy").astype(np.float64)
        i8 = np.load(f"{R}/emb_int8/{ds}/{m}_int8__clean.npy").astype(np.float64)
        r = {"n": fp.shape[0], "dim": fp.shape[1]}
        def stats(x, name):
            d = {}
            d["rel_l2_mean"] = float(np.mean(np.linalg.norm(x - fp, axis=1) / np.linalg.norm(fp, axis=1)))
            d["maxabs"] = float(np.abs(x - fp).max())
            d["cos_mean"] = float(np.mean(np.sum(x * fp, 1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(fp, axis=1))))
            d["norm_ratio_mean"] = float(np.mean(np.linalg.norm(x, axis=1) / np.linalg.norm(fp, axis=1)))
            # per-dimension affine fit x_d = a_d * fp_d + b_d
            fm, xm = fp.mean(0), x.mean(0)
            cov = ((fp - fm) * (x - xm)).mean(0); var = ((fp - fm) ** 2).mean(0)
            a = cov / np.maximum(var, 1e-12); b = xm - a * fm
            resid = x - (a * fp + b)
            d["affine_perdim_resid_rel"] = float(np.linalg.norm(resid) / np.linalg.norm(x - fp))
            d["affine_perdim_R2"] = float(1 - (resid ** 2).sum() / ((x - xm) ** 2).sum())
            # single global scale
            g = float((x * fp).sum() / (fp * fp).sum())
            d["global_scale"] = g
            d["global_scale_resid_rel"] = float(np.linalg.norm(x - g * fp) / np.linalg.norm(x - fp))
            # per-dimension correlation
            c = cov / np.sqrt(np.maximum(var, 1e-12) * np.maximum(((x - xm) ** 2).mean(0), 1e-12))
            d["perdim_corr_median"] = float(np.median(c)); d["perdim_corr_p10"] = float(np.percentile(c, 10))
            d["perdim_corr_min"] = float(c.min()); d["n_dims_corr_lt_0.5"] = int((c < 0.5).sum())
            d["perdim_scale_a_p10_p50_p90"] = [float(v) for v in np.percentile(a, [10, 50, 90])]
            # zero-variance / dead dimensions
            d["n_dims_const_in_x"] = int((x.std(0) < 1e-6).sum()); d["n_dims_const_in_fp32"] = int((fp.std(0) < 1e-6).sum())
            # standardized-space error: mean/std from FP32 (as the head's scaler does)
            mu, sd = fp.mean(0), fp.std(0) + 1e-12
            z_fp, z_x = (fp - mu) / sd, (x - mu) / sd
            d["z_space_rel_l2_mean"] = float(np.mean(np.linalg.norm(z_x - z_fp, axis=1) / np.linalg.norm(z_fp, axis=1)))
            d["z_space_meanabs_shift_per_dim_median"] = float(np.median(np.abs((xm - mu) / sd)))
            d["z_space_meanabs_shift_per_dim_max"] = float(np.abs((xm - mu) / sd).max())
            d["z_space_n_dims_shift_gt_1sd"] = int((np.abs((xm - mu) / sd) > 1).sum())
            r[name] = d
        stats(o32, "ortfp32"); stats(i8, "int8"); stats(w8, "w8a8")
        mu, sd = fp.mean(0), fp.std(0)
        shift = np.abs(i8.mean(0) - mu) / (sd + 1e-12)
        top = np.argsort(-shift)[:8]
        r["int8_worst_dims"] = [{"dim": int(d), "shift_sd": float(shift[d]), "fp32_mean": float(mu[d]), "fp32_std": float(sd[d]),
                                 "int8_mean": float(i8[:, d].mean()), "int8_std": float(i8[:, d].std()), "int8_n_unique": int(len(np.unique(i8[:, d]))),
                                 "w8a8_mean": float(w8[:, d].mean()), "ortfp32_mean": float(o32[:, d].mean())} for d in top]
        r["fp32_std_percentiles_1_10_50"] = [float(v) for v in np.percentile(sd, [1, 10, 50])]
        r["n_dims_fp32_std_lt_1e-3"] = int((sd < 1e-3).sum()); r["n_dims_fp32_std_lt_1e-2"] = int((sd < 1e-2).sum())
        # int8 output grid: smallest positive gap between unique values
        u = np.unique(i8); gaps = np.diff(u); r["int8_min_value_gap"] = float(gaps[gaps > 1e-9].min()); r["int8_n_unique_values"] = int(len(u))
        # also int8 vs w8a8 directly
        r["int8_vs_w8a8_rel_l2_mean"] = float(np.mean(np.linalg.norm(i8 - w8, axis=1) / np.linalg.norm(w8, axis=1)))
        r["fp32_range"] = [float(fp.min()), float(fp.max())]; r["int8_range"] = [float(i8.min()), float(i8.max())]
        r["fp32_frac_zero"] = float((fp == 0).mean()); r["int8_frac_zero"] = float((i8 == 0).mean())
        res[f"{ds}/{m}"] = r
        print(f"== {ds}/{m}", json.dumps(r, indent=1), flush=True)
json.dump(res, open(out, "w"), indent=1)
