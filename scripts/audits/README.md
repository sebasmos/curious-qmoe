# Audit and fact-check scripts

The code of the separate audit passes and of the fact checks behind a few numbers in the text. Each audit was written
by a different agent from the experiment it checks, with its own analysis code; none imports the experiment's analysis
or cost functions unless its header says so. The scripts are copied as they were run, except that six scripts of
`idea6-stage1-audit/` now read the results root from `RESULTS_DIR` (one added line each). In this archive the machine
paths are rewritten (the results root reads `RESULTS_DIR` or `~/Documents/results`), so a few `run_all.sh` files and
checks that pinned a worktree of the experiment's code now name a folder that does not exist on another machine: set
`W` (or `WT`) to this repository and `R`, `A` to the run folders. Every script reads stored run folders, which are not
in this archive (see section 6 of `REPRODUCE_PAPER.md`).

| Folder | Audits | Numbers in the paper | Entry points |
|---|---|---|---|
| `t21-audit/` | T21 on FSDKaggle2018 (`moe-t21`, -AUDIT) | field and clean gains 2.830 and 2.070 rebuilt; fixed AST cost per clip, +2.61 (p 0.0025); 19 and 187 cross-fold near duplicates, +2.87 without the 187 | `audit_recompute.py <T21 run> <out>`, `audit_costs.py`, `audit_leakage.py`, `audit_sensitivity.py` |
| `t21-audit-d2/` | T21 on FSD50K, and the FSDKaggle2018 background check (-AUDIT-D2) | every FSD50K number within 0.002; exact hull recompute 2.04 (2.05 with absent classes dropped); 1.98 without the two parent classes; +2.86 without the 57 clips whose background is an FSDKaggle2018 recording | `audit_recompute_d2.py`, `audit_sensitivity_d2.py`, `audit_d1_noise_overlap.py`, `audit_dups_d2.py`, `audit_costs_d2.py`, `audit_leakage_d2.py` |
| `idea6-stage3c-audit/` | idea 6 stage 3c (`idea6-stage3c-audit`) | int8 standardiser shift (one MN10 dimension, 147 SD); P3 runner-up 0.07 points; UrbanSound8K re-test +1.29 and exact length +1.37 recomputed | `a1_compare_emb.py`, `a2_scaler_test.py`, `c2_p3_recompute.py`, `c3_p5b_retrained.py` |
| `idea5-stage3-audit/` | idea 5 stage 2 against time-sharing (`idea5-stage3-audit`) | hull better in 52 of 52 cells, by 0.13 to 10.2 | `run_all.sh` |
| `t19-audit/` | T19 (`moe-t19`, -AUDIT) | H1 and H2 rebuilt | `reproduce_h1_h2.py`, `check_*.py` |
| `idea6-stage1-audit/` | idea 6 stage 1 | stage 1 pass rebuilt | `run_all.sh` |
| `idea6-stage2-audit/` | idea 6 stage 2 | stage 2 passes rebuilt | `run_all.sh`, `run_recompute.sh`, `run_humility.sh` |
| `fact-checks/` | checks run while answering external reviews of an earlier draft | heads: fold-level equivalence at a one-point margin, p at most 0.043 (`table1_fold_level.py`); AST-Q4 and AST-Q4ch under white noise at 0 dB on ESC-50 (50 classes), 1.6 and 11.7 points (`train_w4_heads.sh`, `q4ch_head_transfer.py`, `q4ch_summary.py`); `q4ch_auroc.py` | as listed |
