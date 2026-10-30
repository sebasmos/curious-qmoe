"""T14 extension (ANALYSIS_PLAN_T14.md amendments 1 and 2): the stability measures for every idea and system with stored per-seed, per-fold values, one row each.
Every row carries its protocol; a ratio is only formed inside one protocol and one source. python t14_ideas.py --r RESULTS_DIR"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t14_pair import PAIR, S3C, S2, S3B, Q, FINAL, OPTB, CORR, TASKS
from t14_primary_data import static, single
import t14_eval as E, t14_stats as S
ap = argparse.ArgumentParser(); ap.add_argument("--r", required=True, type=Path); ap.add_argument("--boot", type=int, default=4000); a = ap.parse_args(); R = a.r; S.B = a.boot
BV = json.load(open(R / "inputs/best_single_v2.json")); PF = pd.read_csv(R / "pair/pair_cell_fold.csv"); PV = json.load(open(R / "tables/primary_verdict.json"))
def ser(df, cell="cell", f1="f1", scale=100.0): return pd.DataFrame({"cell": df[cell].to_numpy(), "seed": df["seed"].to_numpy(), "fold": df["fold"].to_numpy(), "f1": df[f1].to_numpy(float) * scale})
SPECS = []; NOTRUN = []
def add(**k): SPECS.append(k)
# ---- reference from the stage 3c pool (priced cost), amended selection on the inner validation of the family -------------------------------------------------------
def pool_ref(task, cells, cost):
    key = "additive14" if (task == "esc50" and not any(c in CORR for c in cells)) else "main"
    rc = {}
    def fam(f):
        allsc = BV[task][f"{key}:{f}"]["all_scores"]; ok = [s for s, (v, c) in allsc.items() if c <= cost * (1 + 1e-9)]
        if not ok: ok = sorted(allsc, key=lambda s: allsc[s][1])[:1]
        n = sorted(ok, key=lambda s: (-allsc[s][0], s))[0]; rc[f] = allsc[n][1]; return n, single(task, n)
    refs = {"clean": fam("clean"), "field": fam("field"), "shift": fam("shift")}
    return refs, "inner_validation_same_family", rc
def cascade_rows(path, mask, cond="condition", cost="cost"):
    d = pd.read_csv(path); d = d[mask(d)]; return d
# ---- stage 2 / 3b / option C cascades -----------------------------------------------------------------------------------------------------------------------------
def casc(label, id_, root, fname, task, keys, costcol, proto=3):
    p = Path(root) / f"analysis/{task}/{fname}"
    if not p.exists(): return
    d = pd.read_csv(p); m = np.ones(len(d), bool)
    for k, v in keys.items(): m &= (d[k].astype(str) == v).to_numpy()
    g = d[m]
    if g.empty: NOTRUN.append(dict(id=id_, task=task, reason="pick not found in " + fname)); return
    cells = sorted(g.condition.unique()); cost = float(g[g.condition == "clean"][costcol].mean()) if costcol else np.nan
    refs, sel, rc = pool_ref(task, cells, cost)
    add(id=id_, label=label, protocol=proto, task=task, source=Path(root).name, metric="F1", ser=ser(g, "condition"), ref=refs, ref_selection=sel, cost=cost, note="", ref_cost=rc)
for task in ("esc50", "esc5", "urban"):
    s2 = json.load(open(Q / f"2026-10-22_1120_redesign-idea6-stage2_code-9f346ba/analysis/{task}/picks.json"))["H1"]["design"]
    casc("Idea 6 stage 2 H1 (8-bit EfficientNet to short AST)", "I6b", Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba", "cascades.csv.gz", task,
         dict(cheap_bb=s2[0], cheap=s2[1], expensive=s2[2], rule=s2[3], budget=s2[4]), "cost_priced")
    p2 = json.load(open(Q / f"2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db/analysis/{task}/picks.json"))["P2"]["design"]
    casc("Idea 6 stage 3b P2 (small CNN to AST)", "I6c", Q / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db", "cascades.csv.gz", task,
         dict(cheap_src=p2[0], cheap_bb=p2[1], cheap=p2[2], expensive=p2[3], rule=p2[4], budget=p2[5]), "cost")
    pc = json.load(open(Q / f"2026-10-23_2133_redesign-idea6-optionC_code-d6e7913/analysis/{task}/picks.json"))
    h = pc["H1"]["design"]; casc("Option C H1 (two tiers)", "OC-H1", Q / "2026-10-23_2133_redesign-idea6-optionC_code-d6e7913", "cascades.csv.gz", task,
                       dict(cheap_src=h[0], cheap_bb=h[1], cheap=h[2], expensive=h[3], rule=h[4], budget=h[5]), "cost")
    e = pc["E1"]["design"]
    casc("Option C E1 (three tiers)", "OC-E1", Q / "2026-10-23_2133_redesign-idea6-optionC_code-d6e7913", "cascades3.csv.gz", task,
         dict(cheap_src=e[0], cheap_bb=e[1], cheap=e[2], mid=e[3], top="w8g64@512" if task == "esc50" else None or "x", rule=e[4], budget=e[5], b2=e[6]), "cost") if False else None
# three-tier option C E1: top tier column is stored; use the pick's own (mid) and any top present
for task in ("esc50", "esc5", "urban"):
    pc = json.load(open(Q / f"2026-10-23_2133_redesign-idea6-optionC_code-d6e7913/analysis/{task}/picks.json")); e = pc["E1"]["design"]
    d = pd.read_csv(Q / f"2026-10-23_2133_redesign-idea6-optionC_code-d6e7913/analysis/{task}/cascades3.csv.gz")
    m = (d.cheap_src == e[0]) & (d.cheap_bb == e[1]) & (d.cheap.astype(str) == e[2]) & (d.mid == e[3]) & (d.rule == e[4]) & (d.budget == e[5]) & (d.b2.astype(str) == e[6])
    g = d[m]
    if g.empty or g.top.nunique() != 1: NOTRUN.append(dict(id="OC-E1", task=task, reason=f"E1 pick matches {g.top.nunique() if len(g) else 0} top tiers")); continue
    cost = float(g[g.condition == "clean"].cost.mean()); refs, sel, rc = pool_ref(task, sorted(g.condition.unique()), cost)
    add(id="OC-E1", label="Option C E1 (three tiers)", protocol=3, task=task, source="optionC", metric="F1", ser=ser(g, "condition"), ref=refs, ref_selection=sel, cost=cost, note="", ref_cost=rc)
# ---- stage 1 (clean), idea 5 stage 2 --------------------------------------------------------------------------------------------------------------------------------
for task in ("esc50", "esc5", "urban"):
    root = Q / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260/analysis"; pf = pd.read_csv(root / "primary_fp32.csv"); r = pf[pf.task == task].iloc[0]
    c = pd.read_csv(root / task / "cascades.csv"); g = c[(c.cheap.astype(str) == str(r.cheap)) & (c.expensive == r.expensive) & (c.rule == r.rule) & (c.point == r.point)]
    st = pd.read_csv(root / task / "static.csv"); cost = float(r.cost)
    mean = st.groupby("system").agg(f1=("f1", "mean"), cost=("cost_fp32", "mean")); ok = mean[mean.cost <= cost * (1 + 1e-9)]; nm = ok.f1.idxmax()
    ref = pd.DataFrame({"cell": "clean", "seed": st[st.system == nm].seed, "fold": st[st.system == nm].fold, "f1": 100 * st[st.system == nm].f1})
    add(id="I6a", label="Idea 6 stage 1 (EfficientNet to AST, clean)", protocol=3, task=task, source="idea6-stage1", metric="F1", ser=ser(g.assign(cell="clean"), "cell"),
        ref={f: (nm, ref) for f in ("clean", "field", "shift")}, ref_selection="test_oracle", cost=cost, note="clean only; unpriced FP32 cost")
for task in ("esc50", "esc5", "urban"):
    root = Q / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/analysis" / task
    c = pd.read_csv(root / "cascades.csv"); g = c[(c.cheap == "n3g64") & (c.expensive == "w8g64") & (c["head"].astype(str) == "esc") & (c.signal == "margin") & (c.point == "b05")]
    st = pd.read_csv(root / "static.csv"); st = st[st["head"].astype(str) == "esc"]
    bit = st[st.condition == "clean"].groupby("config").bitops.mean()
    rate = float(g[g.condition == "clean"].rate.mean()); cost = float(bit["n3g64"] + rate * bit["w8g64"]); cells = sorted(g.condition.unique())
    mean = st[st.condition == "clean"].groupby("config").f1.mean(); ok = [k for k in mean.index if bit[k] <= cost * (1 + 1e-9)]
    nm = max(ok, key=lambda k: mean[k]) if ok else "n3g64"
    ref = ser(st[st.config == nm], "condition"); ref = ref[ref.cell.isin(cells)]
    add(id="I5", label="Idea 5 stage 2 (AST n3g64 to w8g64, FP32 head, margin, b05)", protocol=3, task=task, source="idea5-stage2", metric="F1", ser=ser(g, "condition"),
        ref={f: (nm, ref) for f in ("clean", "field", "shift")}, ref_selection="test_oracle", cost=cost, note="bit-ops of the source (unpriced)")
# ---- idea 7 -------------------------------------------------------------------------------------------------------------------------------------------------------
for task in ("esc50", "esc5", "urban"):
    d = pd.read_csv(Q / f"2026-10-22_1105_redesign-idea7-stage1_code-e2af723/outputs/{task}/pairs.csv.gz"); g = d[(d.device == "n3g64") & (d["head"].astype(str) == "esc") & (d.rule == "pre")]
    if g.empty: NOTRUN.append(dict(id="I7", task=task, reason="primary row missing")); continue
    dv = pd.read_csv(Q / f"2026-10-22_1105_redesign-idea7-stage1_code-e2af723/outputs/{task}/device.csv.gz")
    dd = dv[(dv.device == "n3g64") & (dv["head"].astype(str) == "esc")] if {"device", "head"} <= set(dv.columns) else dv
    ref = ser(dd, "condition", "f1_device") if "f1_device" in dd.columns else None
    if ref is None or ref.empty: NOTRUN.append(dict(id="I7", task=task, reason="device-alone series missing")); continue
    ref = ref.groupby(["cell", "seed", "fold"], as_index=False).f1.mean()
    add(id="I7", label="Idea 7 device to cloud (3-bit device, FP32 head, preset rate)", protocol=3, task=task, source="idea7-stage1", metric="F1", ser=ser(g, "condition", "f1_sys"),
        ref={f: ("device alone (n3g64)", ref) for f in ("clean", "field", "shift")}, ref_selection="device alone (cheaper single)", cost=np.nan, note="reference is the device model alone")
# ---- T13 -----------------------------------------------------------------------------------------------------------------------------------------------------------
T13 = Q / "2026-10-25_0145_moe-t13_code-eaaedc9"; cf = pd.read_csv(T13 / "cell_fold_acc.csv"); pre = json.load(open(Path(__file__).resolve().parent / "t13_preselected.json"))
for task in ("esc50", "esc5", "urban", "tau"):
    pr = pre[task]["pair"]; bs = pre[task]["best_single"]
    g = cf[(cf.task == task) & (cf.system == pr) & (cf.kind == "mean")]; r = cf[(cf.task == task) & (cf.system == bs) & (cf.kind == "single")]
    if g.empty or r.empty: NOTRUN.append(dict(id="T13", task=task, reason="pair or single not in cell_fold_acc")); continue
    cells = sorted(set(g.cell) & set(r.cell)); g, r = g[g.cell.isin(cells)], r[r.cell.isin(cells)]; cst = float(pre[task].get("cost", np.nan))
    if task != "tau":
        refs, sel, rc = pool_ref(task, cells, cst)
        add(id="T13", label=f"T13 uniform mixture {pr}", protocol=3, task=task, source="moe-t13", metric="F1", ser=ser(g, scale=1.0), ref=refs, ref_selection=sel, cost=cst, note="reference re-selected on the stage 3c pool (amendment 2); T13's own clean-accuracy pick was " + bs, ref_cost=rc)
    else:
        add(id="T13", label=f"T13 uniform mixture {pr}", protocol=3, task=task, source="moe-t13", metric="F1", ser=ser(g, scale=1.0), ref={f: (bs, ser(r, scale=1.0)) for f in ("clean", "field", "shift")},
            ref_selection="clean_validation_accuracy (T13 pre-selection; no TAU pool)", cost=cst, note="TAU: clean and shift cells only")
# ---- T2b, T7, T1 ---------------------------------------------------------------------------------------------------------------------------------------------------
t2 = pd.read_csv(Q / "2026-10-24_2224_moe-t2b_code-f5e4935/urban/t2b_per_seed_fold.csv")
ARMS = {"A1": "hard margin threshold router", "A2h": "hard router (cheap features)", "A3h": "soft mixture (cheap gate)", "A3f": "soft mixture (full features)", "A4": "always-average", "A5": "stacking"}
for arm, nm in ARMS.items():
    g = t2[t2.arm == arm].assign(cell="clean"); r = t2[t2.arm == "BEST"].assign(cell="clean")
    if g.empty: continue
    add(id=f"T2b-{arm}", label=f"T2b {arm} {nm}", protocol=3, task="urban", source="moe-t2b", metric="F1", ser=ser(g, scale=1.0), ref={f: ("BEST single expert (clean out-of-fold)", ser(r, scale=1.0)) for f in ("clean", "field", "shift")},
        ref_selection="clean_out_of_fold_accuracy", cost=float(g.cost.mean()), note="clean only; T2b plan's own best single")
t2c = pd.read_csv(Q / "2026-10-25_0052_moe-t1c_code-7d7429e/t2b/t2b_per_seed_fold.csv"); CM = {"U-field": "field", "E-field": "field", "T-shift": "shift"}
for bed, task in (("urban", "urban"), ("esc50", "esc50"), ("tau", "tau")):
    for arm, nm in ARMS.items():
        g = t2c[(t2c.bed == bed) & (t2c.arm == arm) & t2c.cond.isin(CM)]; r = t2c[(t2c.bed == bed) & (t2c.arm == "BEST") & t2c.cond.isin(CM)]
        if g.empty: continue
        add(id=f"T2b-T1c-{arm}", label=f"T2b (T1c beds) {arm} {nm}", protocol=3, task=task, source="moe-t1c/t2b", metric="F1", ser=ser(g.assign(cell=g.cond.map(CM)), scale=1.0),
            ref={f: ("BEST single expert (clean out-of-fold)", ser(r.assign(cell=r.cond.map(CM)), scale=1.0)) for f in ("clean", "field", "shift")}, ref_selection="clean_out_of_fold_accuracy", cost=float(g.cost.mean()), note="field mix / TAU shift only")
for task in ("urban", "esc50"):
    t7 = pd.read_csv(Q / f"2026-10-24_2313_moe-t7_code-9fb5b35/{task}/arms_per_seed_fold.csv")
    if "domain" not in t7: t7["domain"] = "clean"
    arm7 = "CE_energy25" if (t7.arm == "CE_energy25").any() else "LIB_energy25"; g = t7[t7.arm == arm7]; r = t7[t7.arm == "E1"]
    add(id="T7", label=f"T7 masked expert, energy 25 percent ({arm7})", protocol=3, task=task, source="moe-t7", metric="F1", ser=ser(g.assign(cell=g.domain)), ref={f: ("E1 cheap model (only cheaper single)", ser(r.assign(cell=r.domain))) for f in ("clean", "field", "shift")},
        ref_selection="only single of lower cost", cost=np.nan, note="the full 4-bit AST costs more")
t1 = pd.read_csv(Q / "2026-10-25_0052_moe-t1c_code-7d7429e/t1/t1c_arm_metrics_per_seed_fold.csv")
for bed, task in (("urban", "urban"), ("esc50", "esc50"), ("tau", "tau")):
    g = t1[(t1.bed == bed) & t1.cond.isin(CM)]
    if g.empty: continue
    add(id="T1c", label="T1c specialist R10 (accuracy, not F1)", protocol=3, task=task, source="moe-t1c", metric="accuracy", ser=ser(g.assign(cell=g.cond.map(CM)), "cell", "R10_all", 1.0),
        ref={f: ("uniform twin U (same cost)", ser(g.assign(cell=g.cond.map(CM)), "cell", "U_all", 1.0)) for f in ("clean", "field", "shift")}, ref_selection="uniform twin", cost=np.nan, note="accuracy over all clips; reference is the twin, not a best single")
# ---- protocol 2: original MoE arms and idea 4 ------------------------------------------------------------------------------------------------------------------------
P2 = Q / "2026-10-20_1300_protocol-v2_code-5db7e75/outputs"; P2F = Q / "2026-10-20_1801_protocol-v2-fp32-expert_code-5db7e75/outputs"
def p2_load(base, tag, moe=True):
    rows = []
    for d in sorted(glob.glob(str(base / f"{tag}_s*"))):
        seed = int(d.rsplit("_s", 1)[1])
        for f in glob.glob(f"{d}/{'moe/' if moe else ''}fold_*/metrics.json") if moe else []:
            m = json.load(open(f)); rows.append(dict(cell="clean", seed=seed, fold=int(m["fold"]), f1=m["f1_macro"], ivf1=100 * m["inner_val_f1_macro"], flops=m["validation_flops"]))
    return pd.DataFrame(rows)
def p2_single(base, task, kind):
    rows = []
    for d in sorted(glob.glob(str(base / f"{task}_singles_s*"))):
        seed = int(d.rsplit("_s", 1)[1])
        for f in glob.glob(f"{d}/{kind}/fold_*/metrics.json"):
            m = json.load(open(f)); rows.append(dict(cell="clean", seed=seed, fold=int(m["fold"]), f1=m["f1_macro"], ivf1=100 * m["inner_val_f1_macro"], flops=m["validation_flops"]))
    return pd.DataFrame(rows)
KINDS = ["1", "2", "4", "8", "16", "bitnet", "esc", "qesc"]
for task in ("esc", "urban", "quinn"):
    singles = {k: p2_single(P2, task, k) for k in KINDS}; singles = {k: v for k, v in singles.items() if len(v)}
    tags = [("pprior_a1.0", "Original MoE, precision-prior router (a1.0)"), ("uniform_ctrl", "Old uniform-routing MoE (control)"), ("sharedtrunk_a1.0", "Shared-trunk MoE, precision prior"), ("sharedtrunk_uniform_ctrl", "Shared-trunk MoE, uniform control")]
    base = {t: p2_load(P2, f"{task}_{t}") for t, _ in tags}
    for t, lab in tags:
        g = base[t]
        if g.empty: continue
        fl = g.flops.mean(); ok = {k: v for k, v in singles.items() if v.flops.mean() <= fl * (1 + 1e-9)} or {k: v for k, v in list(singles.items())[:1]}
        kb = max(ok, key=lambda k: ok[k].ivf1.mean()); ctrl = base["uniform_ctrl"] if t == "pprior_a1.0" else (base["sharedtrunk_uniform_ctrl"] if t == "sharedtrunk_a1.0" else None)
        add(id=f"P2-{t}", label=lab, protocol=2, task="esc5" if task == "esc" else task, source="protocol-v2 (leaky folds)", metric="F1", ser=ser(g), ref={f: (f"protocol-2 single head '{kb}'", ser(singles[kb])) for f in ("clean", "field", "shift")},
            ref_selection="inner_validation_f1 (protocol 2)", cost=float(fl), note="clean only; protocol-2 task 'esc' is the five-group ESC task", ctrl=(ser(ctrl) if ctrl is not None and len(ctrl) else None), ctrl_name="uniform control (same source)")
if (P2F).exists():
    for t in ("pprior_a1.0", "pprior_a0.3", "uniform_ctrl"):
        g = p2_load(P2F, f"esc_fp32top_{t}")
        if len(g):
            singles = {k: p2_single(P2, "esc", k) for k in KINDS}; fl = g.flops.mean(); ok = {k: v for k, v in singles.items() if len(v) and v.flops.mean() <= fl * (1 + 1e-9)} or {"esc": singles["esc"]}
            kb = max(ok, key=lambda k: ok[k].ivf1.mean()); ctrl = p2_load(P2F, "esc_fp32top_uniform_ctrl")
            add(id=f"P2-fp32top-{t}", label=f"FP32-top-expert MoE {t}", protocol=2, task="esc5", source="protocol-v2-fp32-expert (leaky folds)", metric="F1", ser=ser(g),
                ref={f: (f"protocol-2 single head '{kb}'", ser(singles[kb])) for f in ("clean", "field", "shift")}, ref_selection="inner_validation_f1 (protocol 2)", cost=float(fl), note="clean only",
                ctrl=(ser(ctrl) if t != "uniform_ctrl" else None), ctrl_name="uniform control (same source)")
d4 = json.load(open(Q / "2026-10-20_2039_redesign-idea4-stage1_code-5ecd753/analysis/diag_esc_s0.json"))
ctrl4 = pd.DataFrame([dict(cell="clean", seed=0, fold=f["fold"], f1=100 * f["f1_macro"]) for f in d4["uniform_ctrl"]["folds"]])
_s0 = {k: p2_single(P2, "esc", k) for k in KINDS}; _s0 = {k: v[v.seed == 0] for k, v in _s0.items() if len(v)}; _fl = p2_load(P2, "esc_pprior_a1.0").flops.mean()
_ok = {k: v for k, v in _s0.items() if v.flops.mean() <= _fl * (1 + 1e-9)}; _kb = max(_ok, key=lambda k: _ok[k].ivf1.mean())
for arm in ("pprior_a1.0", "A_dense"):
    g = pd.DataFrame([dict(cell="clean", seed=0, fold=f["fold"], f1=100 * f["f1_macro"]) for f in d4[arm]["folds"]])
    add(id=f"I4-{arm}", label=f"Idea 4 repaired router, arm {arm}", protocol=2, task="esc5", source="idea4-stage1 (protocol 2, seed 0)", metric="F1", ser=g,
        ref={f: (f"protocol-2 single head '{_kb}' seed 0", ser(_s0[_kb])) for f in ("clean", "field", "shift")}, ref_selection="inner_validation_f1 (protocol 2)", cost=float(_fl), note="one seed, clean",
        ctrl=ctrl4, ctrl_name="uniform control (same source)")
# V4 claim
v4 = json.load(open(Q / "2026-10-23_1146_reaudit-arxiv-v4_B/results/claimEJ_stats.json"))["E"]
def arr(k): return pd.DataFrame(dict(cell="clean", seed=0, fold=range(1, 6), f1=100 * np.array(v4[k])))
for k in ("r1", "r2"):
    add(id=f"V4-{k}", label=f"Original paper v4, precision-prior router run {k} (one 5-fold run per arm)", protocol="2/old", task="esc5", source="reaudit-arxiv-v4_B claimEJ_stats.json", metric="F1", ser=arr(k), ref={}, ref_selection="none", cost=np.nan,
        note="old protocol, leaky folds; the baseline of the '68 to 97%' claim", ctrl=arr("base"), ctrl_name="uniform baseline run (same source)")
# idea 1 stage 2
rec = pd.read_json(Q / "2026-10-21_0002_redesign-idea1-stage2_code-85e912a/analysis/main/records.jsonl", lines=True)
for ds, task in (("esc50", "esc5"), ("urban", "urban"), ("quinn", "quinn")):
    r = rec[rec.dataset == ds]; g = r[(r.method == "c1to8") & (r.operating_point == "b05")]
    if g.empty: continue
    sg = r[r.is_single]; cost = float(g.bitops_per_clip.mean()); m = sg.groupby("method").agg(f1=("f1_macro", "mean"), c=("bitops_per_clip", "mean")); ok = m[m.c <= cost * (1 + 1e-9)]; nm = (ok if len(ok) else m).f1.idxmax()
    add(id="I1", label="Idea 1 head cascade c1to8 b05 (stage 2)", protocol=3, task=task, source="idea1-stage2", metric="F1", ser=ser(g.assign(cell="clean"), "cell", "f1_macro"),
        ref={f: (f"single head '{nm}'", ser(sg[sg.method == nm].assign(cell="clean"), "cell", "f1_macro")) for f in ("clean", "field", "shift")}, ref_selection="test_oracle", cost=cost, note="clean only; protocol-3 heads")
# ---- the primary pair and its comparators (protocol 3), from the primary analysis ----------------------------------------------------------------------------------
for task in PAIR:
    cells = PV[task]["cells"]; ref = {"clean": (PV[task]["best_single_clean"], single(task, PV[task]["best_single_clean"])), "field": (PV[task]["best_single_field"], single(task, PV[task]["best_single_field"])),
                                      "shift": (PV[task]["best_single_shift"], single(task, PV[task]["best_single_shift"]))}
    for col, id_, lab in (("pair", "I6d", "Margin-routed pair (stage 3c P3)"), ("avg", "AA", "Always-average of the pair's two experts"), ("cheap", "I6d-cheap", "Pair's cheap expert alone"), ("strong", "I6d-strong", "Pair's strong expert alone")):
        g = PF[PF.task == task]; add(id=id_, label=lab, protocol=3, task=task, source="moe-t14 pair", metric="F1", ser=g[["cell", "seed", "fold"]].assign(f1=g[col].to_numpy()), ref=ref, ref_selection="inner_validation_same_family", cost=np.nan, note="")
# T4, information-based routing losses, ideas 2/3, T9, T12
for i_, why in (("T4", "RL router control: planned, never built (T2 to T6 skipped by the stop rule)"), ("IBL", "information-based routing losses: planned, never built"),
                ("I2", "idea 2 learned deferral: seed 0 only, aggregates only for the headline: not determinable (one seed)"), ("I3", "idea 3 disagreement: logits per fold for one or two seeds, clean only, no per-fold F1 table, no pick: not computed"),
                ("I1-stage1", "idea 1 stage 1: logits only, 1 to 2 seeds, protocol 2: not computed"), ("T9", "any-precision tier selection: per-fold differences to the hull only, no per-seed F1: not determinable"),
                ("T12", "adapter expert: micro-pilot, fold 1 only, flagged-clip accuracy: not determinable"), ("I5b", "idea 5b hull of random mixing: no pre-registered pick (post hoc best points only): not computed")):
    NOTRUN.append(dict(id=i_, task="all", reason=why))
# ---- evaluation ------------------------------------------------------------------------------------------------------------------------------------------------------
rows = []
def lab_sampling(ls):
    ls = [l for l in ls if l]
    if not ls: return "not determinable at five folds"
    if all(l == "more stable" for l in ls): return "more stable"
    if all(l == "less stable" for l in ls): return "less stable"
    if all(l == "as stable" for l in ls): return "as stable"
    return "not determinable at five folds"
for sp in SPECS:
    s_ = sp["ser"]
    if s_.empty: NOTRUN.append(dict(id=sp["id"], task=sp["task"], reason="empty series (pick or source rows not found)")); continue
    rcs = sp.get("ref_cost", {}); row = dict(ref_cost_shift=rcs.get("shift", np.nan), ref_cost_clean=rcs.get("clean", np.nan), id=sp["id"], system=sp["label"], protocol=sp["protocol"], task=sp["task"], source=sp["source"], metric=sp["metric"], ref_selection=sp["ref_selection"], cost=sp["cost"], note=sp["note"])
    X = E.mat(s_, s_.cell.iloc[0]); row.update(n_seed=X.shape[0], n_fold=X.shape[1], cells=";".join(sorted(s_.cell.unique())))
    labs = []; 
    for cond in ("clean", "field", "shift"):
        d = E.descr(s_, cond)
        if not d: continue
        row.update({f"{cond}_mean": d["mean"], f"{cond}_sd_fold": d["sd_fold"], f"{cond}_sd_seed": d["sd_seed"], f"{cond}_sd_inter": d["sd_inter"]})
        if sp["ref"]:
            nm, rf = sp["ref"][cond] if cond in sp["ref"] else (None, None)
            if rf is not None and not rf[rf.cell == cond].empty:
                sm = E.sampling(s_, rf, cond)
                if sm:
                    row.update({f"{cond}_ref": nm, f"{cond}_ref_mean": E.descr(rf, cond)["mean"], f"{cond}_ref_sd_fold": E.descr(rf, cond)["sd_fold"], f"{cond}_ratio": sm["ratio_fold"], f"{cond}_ratio_lo": sm["ratio_fold_lo"], f"{cond}_ratio_hi": sm["ratio_fold_hi"],
                                f"{cond}_ratio_seed": sm["ratio_seed"], f"{cond}_ratio_inter": sm["ratio_inter"], f"{cond}_bf_p": sm["bf_p"], f"{cond}_label": sm["sampling_label"]}); labs.append(sm["sampling_label"])
        if sp.get("ctrl") is not None:
            sm = E.sampling(s_, sp["ctrl"], cond)
            if sm: row.update({f"{cond}_ratio_vs_control": sm["ratio_fold"], f"{cond}_ratio_vs_control_lo": sm["ratio_fold_lo"], f"{cond}_ratio_vs_control_hi": sm["ratio_fold_hi"], f"{cond}_bf_p_vs_control": sm["bf_p"], "control": sp["ctrl_name"]})
    if not sp["ref"]:
        cl = [E.sampling(s_, sp["ctrl"], c) for c in ("clean", "field") if sp.get("ctrl") is not None and E.mat(s_, c) is not None]
        labs = [x["sampling_label"] for x in cl if x]; row["sampling_label"] = lab_sampling(labs) + " (against the same source's uniform control, no best single)"
    else: row["sampling_label"] = lab_sampling(labs)
    if sp["ref"] and "shift" in sp["ref"]:
        nm, rf = sp["ref"]["shift"]; nbr = 1.0 / (row["n_fold"] - 1); sh = E.shift(s_, rf, nbr)
        if sh: row.update({f"shift_{k}": v for k, v in sh.items()}); row["shift_ref"] = nm
    rows.append(row)
df = pd.DataFrame(rows)
# Holm over a system's tasks for the shift label (within system id), and over all rows (extension)
df["shift_label"] = "not scored (fewer than 3 shared cells)"
for id_, g in df[df.shift_dworst.notna()].groupby("id"):
    pw, pd_ = S.holm(g.shift_dworst_p.to_numpy()), S.holm(g.shift_ddrop_p.to_numpy())
    for (i, r), a_, b_ in zip(g.iterrows(), pw, pd_):
        df.loc[i, "shift_holm_worst"], df.loc[i, "shift_holm_drop"] = a_, b_
        df.loc[i, "shift_label"] = "more stable" if (r.shift_dworst >= 0.5 and r.shift_ddrop >= 0.5 and a_ < 0.05 and b_ < 0.05) else "less stable" if (r.shift_dworst <= -0.5 and r.shift_ddrop <= -0.5 and a_ < 0.05 and b_ < 0.05) else "no difference shown"
allp = df[["shift_dworst_p", "shift_ddrop_p"]].stack().dropna()
df["holm_x15_shift"] = np.minimum(1, df[["shift_holm_worst", "shift_holm_drop"]].max(axis=1) * 15)
def overall(g):
    n = len(g); need = min(2, n)
    mo = (g.sampling_label.str.startswith("more stable") | (g.shift_label == "more stable")).sum(); le = (g.sampling_label.str.startswith("less stable") | (g.shift_label == "less stable")).sum()
    if mo >= need and le == 0: return "more stable"
    if le >= need and mo == 0: return "less stable"
    if mo and le: return "mixed"
    if g.sampling_label.str.startswith("as stable").all() and not (g.shift_label.isin(["more stable", "less stable"])).any(): return "as stable"
    return "not determinable at five folds"
df["overall_verdict"] = df.groupby("id").id.transform(lambda s: overall(df.loc[s.index]))
df.loc[df.sampling_label.str.contains("no best single"), "overall_verdict"] = "against control only: " + df.loc[df.sampling_label.str.contains("no best single"), "sampling_label"].str.replace(" (against the same source's uniform control, no best single)", "", regex=False)
df["cost_ratio_vs_ref_clean"] = df.cost / df.ref_cost_clean
KIND = {"I7": "device alone (cheaper single)", "T7": "cheaper single only", "T1c": "uniform twin", "V4-r1": "uniform baseline run only", "V4-r2": "uniform baseline run only"}
df["reference_kind"] = [KIND.get(i, "best single, test oracle" if "test_oracle" in str(rs) else "best single, clean out-of-fold" if "out_of_fold" in str(rs) else "best single, inner validation") for i, rs in zip(df.id, df.ref_selection)]
m_ = df.id.isin(["I7", "T7", "T1c"]); df.loc[m_, "overall_verdict"] = "vs " + df.loc[m_, "reference_kind"] + ": " + df.loc[m_, "overall_verdict"]
df.to_csv(R / "tables/stability_rows.csv", index=False); pd.DataFrame(NOTRUN).to_csv(R / "tables/not_computed.csv", index=False)
print(df[["id", "protocol", "task", "n_seed", "n_fold", "clean_ratio", "field_ratio", "shift_dworst", "shift_ddrop", "sampling_label", "shift_label", "overall_verdict"]].round(3).to_string())
print(pd.DataFrame(NOTRUN).to_string())
