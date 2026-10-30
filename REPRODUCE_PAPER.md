# Reproducing the paper

Every result in the paper comes from one run folder. This file gives, for each result, the code that produced it,
the commands, the datasets with their checksums, and the time the original run took.

**Runtimes.** All runs used one Apple M2 Pro laptop (12 CPU cores, 32 GB, macOS). Feature extraction ran on the
Apple GPU (PyTorch MPS), and heads, ONNX Runtime, scoring and statistics ran on the CPU at `nice 10` with 2 threads per
job. The times below are the observed wall clock of the original runs. They include waits behind other jobs sharing
the machine, so a dedicated run is usually shorter. "Analysis only" is the time from stored scores to the tables.
Software: macOS 26.7.1 (arm64), Python 3.11.14 with PyTorch 2.7.0, torchaudio 2.7.0, transformers 4.57.6, NumPy 2.2.5,
SciPy 1.14.1, scikit-learn 1.6.1 and pandas 2.2.3; ONNX Runtime 1.31.0 on Python 3.11.17 in a second environment. Every
package version of both environments is in `environment/qmoe.lock.txt` and `environment/qmoe-ort.lock.txt`.

## 1. Setup

```bash
conda create -n qmoe python=3.11 -y && conda activate qmoe && pip install -r environment/qmoe.lock.txt
conda create -n qmoe-ort python=3.11 -y && conda run -n qmoe-ort pip install -r environment/qmoe-ort.lock.txt
export PY=python PYO=$(conda run -n qmoe-ort which python)
export RESULTS_DIR=~/Documents/results     # one folder per run; every driver and analysis script reads this variable
bash scripts/setup_original_layout.sh      # ~/Documents/{qmoe-work,DATASETS,qmoe-weights} and $RESULTS_DIR
bash scripts/download_datasets.sh esc50    # then urban8k, fsdk2018, fsd50k-meta, weights (section 2)
export HF_HUB_OFFLINE=1                    # after the AST has been fetched once
PYTHONPATH=. pytest tests/
```

**Folder conventions.** The drivers and analysis scripts expect the datasets under `~/Documents/DATASETS`, the weights
under `~/Documents/qmoe-weights`, and each run in `$RESULTS_DIR/<run folder>` (default `~/Documents/results` when the
variable is unset). They use the run folder names given below: a later stage finds an earlier stage's outputs by that
name under `$RESULTS_DIR`. Most
drivers take `R` (this run's folder), `W` (the repository root, defaults to it), `PY` and `PYO` (the two Python
interpreters) from the environment. `scripts/setup_original_layout.sh` links `~/Documents/qmoe-work/t17` to `t26` to
this repository. It also installs stand-ins for the GPU lock that the original drivers called: the original machine
ran several jobs on one GPU through a queue, and the stand-ins simply run the command.

Every experiment from idea 6 stage 2 on has an `ANALYSIS_PLAN.md` next to its code, and its amendments are dated.
Section 5 lists, for each plan, its commit time, the time of each amendment and the time of the experiment's first
scored output, with the exceptions.

## 2. Datasets and weights

| Data | Source, licence | Files and MD5 | Size | Command |
|---|---|---|---|---|
| ESC-50 | GitHub `karolpiczak/ESC-50` at commit `33c8ce9e`, CC BY-NC 3.0 | `ESC-50-master.zip` 629e8e9ebc1592bcceb06da6bec40275 (the content check is `meta/esc50.csv` 54a0d0055a10bb7df84ad340a148722e and 2,000 clips) | 0.65 GB | `download_datasets.sh esc50` |
| UrbanSound8K | Zenodo 1203745, CC BY-NC 3.0 | `UrbanSound8K.tar.gz` 9aa69802bbf37fb986f71ec1483a196e; `metadata/UrbanSound8K.csv` 5015c37ce7d54c0bb8cd40a0b5ee2074 | 6.0 GB | `download_datasets.sh urban8k` |
| TAU Urban Acoustic Scenes 2020 Mobile, development | Zenodo 3819968, non-commercial research | 16 audio zips, meta and doc; MD5s listed in `scripts/redesign/idea6/optionB/download_tau2020.sh` | 27.4 GB | `bash scripts/redesign/idea6/optionB/download_tau2020.sh` |
| FSDKaggle2018 | Zenodo 2552860, CC BY 4.0 (clips CC BY, CC0, 3 CC BY-NC) | `audio_train.zip` a8b47ff4b52022c178f88fa5c6f080d0, `audio_test.zip` f85c8665c3fc39311ea1b748e194a81d, `meta.zip` f16828ac8be0e5285b9175a12f90d784, `doc.zip` 4f2d1ac88f33a62f9db3108b269ee1b7 | 5 GB | `download_datasets.sh fsdk2018` |
| FSD50K | Zenodo 4060432, CC BY 4.0 (clips CC BY 3.0, CC0, CC BY-NC 3.0, Sampling+) | `ground_truth.zip` ca27382c195e37d2269c4c866dd73485, `metadata.zip` b9ea0c829a411c1d42adb9da539ed237; audio `dev_audio.z01` to `.z05` and `.zip`, `eval_audio.z01` and `.zip`, MD5s in `scripts/redesign/moe/t21/t21_get_d2.sh` | 30 GB download, 26,272 dev and 4,396 eval clips kept | `download_datasets.sh fsd50k-meta`, then `t21_get_d2.sh` (section 3.1) |
| FSDnoisy18k (T24 D3, T26, T27) | Zenodo 2529934, CC BY 4.0 (clips CC BY, CC0) | `FSDnoisy18k.meta.zip` 96e27a4a63b7a2870522ddcedb5d8296, `FSDnoisy18k.doc.zip` 093a1ca185ec341ca4eac14215e7f96b, `FSDnoisy18k.audio_test.zip` 1ac73d70b4ef3f81900d98c261a832de, `FSDnoisy18k.audio_train.zip` 34dc1d34ca44622af5bf439ceb6f0d55 | 9.5 GB | `R=$R bash scripts/redesign/moe/t24/t24_get_d3.sh meta`, then `audio` (section 3.1.1) |
| TUT Acoustic Scenes 2017 (T24 D4, T26) | Zenodo 400515 (development) and 1040168 (evaluation), academic non-commercial | meta, doc and error zips, 10 development and 4 evaluation audio zips; every MD5 in `scripts/redesign/moe/t24/t24_get_d4.sh` | 14.3 GB | `R=$R bash scripts/redesign/moe/t24/t24_get_d4.sh` (section 3.1.1) |
| EfficientAT MN04, MN05, MN10, MN20, DyMN04 | GitHub `fschmid56/EfficientAT` release v0.0.1, MIT | `mn10_as_mAP_471.pt` cd7ce78dd3faf9b7972252201df29643, `mn20_as_mAP_478.pt` 7c6cfe0db5e5b256a4477559500129f1, `mn04_as_mAP_432.pt` d288913b10fd32c968b3d756236060a6, `mn05_as_mAP_443.pt` 712258b9d276be95421f039061074f87, `dymn04_as.pt` 3b9611cfdb89126f3e9f7453dd908dbd | 0.13 GB | `download_datasets.sh weights` |
| AST | Hugging Face `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`, BSD-3-Clause | fetched by `transformers` at the pinned revision | 0.35 GB | first run online |
| PANNs Wavegram-CNN14 and CNN14 (negative result T23 only) | Zenodo 3987831, CC BY 4.0 | `Wavegram_Cnn14_mAP=0.389.pth` 1e3506ab640371e0b5a417b15fd66d21, `Cnn14_mAP=0.431.pth` 541141fa2ee191a88f24a3219fff024e | 0.65 GB | `download_datasets.sh weights` |

`data/meta/esc-50.csv` (MD5 467cac5cf2ec07121d87f739af821a0b) and `data/meta/urbansound8k.csv`
(f4e691fad4d020b0a46039f4a7c48336) are the clip lists every ESC-50 and UrbanSound8K script reads. They hold the
official folds, the label, and `class_id`; for ESC-50, `class_id` is the five fixed groups of ten classes. The
download script copies them to `~/Documents/DATASETS/{ESC-50,Urban8k}/ast_768/`, the paths the drivers use. The
extractors read only the metadata columns. Two steps also need features in a CSV:

- Idea 5 stage 2 (section 3.2, row 1) trains its heads on the FP32 AST embeddings stored in those same CSVs. Rebuild
  them from the clip lists first, which keeps the metadata columns and adds the 768 feature columns:
  `python data/audio_embeddings/extract_ast_embeddings.py ~/Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio data/meta/esc-50.csv ~/Documents/DATASETS/ESC-50/ast_768/esc-50.csv`
  and the same for UrbanSound8K (`.../UrbanSound8K/audio`, `data/meta/urbansound8k.csv`, `.../Urban8k/ast_768/urbansound8k.csv`).
- Ideas 1 to 4 (section 3.5) read the EfficientNet-B3 embedding CSVs that `data/esc/esc-50-data_processing.py` and
  `data/urban8k/urban8k-data_processing.py` write (log-mel images, then the ImageNet EfficientNet-B3 pooled features).

The TAU clip list is built from the dataset's own metadata in section 3.2.

**Pretrained weights.** The files the runs loaded, with the SHA-256 of the copies on the original machine
(`download_datasets.sh weights` checks the MD5):

| Model | URL | File | Bytes | MD5 | SHA-256 |
|---|---|---|---|---|---|
| MN04 | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn04_as_mAP_432.pt | `mn04_as_mAP_432.pt` | 4,066,513 | d288913b10fd32c968b3d756236060a6 | 899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c |
| MN05 | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn05_as_mAP_443.pt | `mn05_as_mAP_443.pt` | 5,878,097 | 712258b9d276be95421f039061074f87 | 97f00bb717ce957dbdf60c26d4de8b161dd7c2593fbe6424859fad7e5569fd7f |
| MN10 | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt | `mn10_as_mAP_471.pt` | 19,708,753 | cd7ce78dd3faf9b7972252201df29643 | 0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e |
| MN20 | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn20_as_mAP_478.pt | `mn20_as_mAP_478.pt` | 71,938,961 | 7c6cfe0db5e5b256a4477559500129f1 | 1561abc3a70373eb4fcae78628d8bde4d64ee7c37f51ba0f4cb89059fcd2a777 |
| DyMN04 | https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/dymn04_as.pt | `dymn04_as.pt` | 8,086,973 | 3b9611cfdb89126f3e9f7453dd908dbd | 49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e |
| AST | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593, revision `f826b80d28226b62986cc218e5cec390b1096902` | `model.safetensors` | 346,404,948 | | ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d |
| AST config | same revision | `config.json`, `preprocessor_config.json` | 26,763; 297 | | a93d525511d77e8ecc933d09674b85099815bbbb417c228a4edd655e252fb9ff; 8d04ba5a9c6fca5d39d0de2b1fd05ecf79deb589fbba279728bbebac39934231 |
| PANNs CNN14 (T15) | Zenodo 3987831 | `Cnn14_mAP=0.431.pth` | 327,428,481 | 541141fa2ee191a88f24a3219fff024e | 0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31 |
| PANNs Wavegram-CNN14 (T23) | Zenodo 3987831 | `Wavegram_Cnn14_mAP=0.389.pth` | 324,062,653 | 1e3506ab640371e0b5a417b15fd66d21 | 7b6553a4f23c48a1b72b532aadfe0ae293c02e5eceb6f4dcddbbaf8906a8055d |

The EfficientAT model code is vendored in `qmoe_lib/efficientat_mn.py` and `efficientat_dymn.py` from EfficientAT
commit `a425fdce92572e602a1d5634799bd9f1f2efa806`; `download_datasets.sh weights` also fetches that commit's
`metadata/class_labels_indices.csv` (MD5 68c0b5c199d6cc789852b144cc3ab500), which `t15_check_cnn14.py` reads.

**Fold lists.** `data/folds/` holds the exact clip lists, with folds and labels, that the runs of the fresh datasets
and TAU read: `fsdkaggle2018/` (T21 D1, `pool.csv` and `extra.csv`), `fsd50k/` (T21 D2), `fsdnoisy18k/` (T24 D3, also
T26 and T27), `tut2017/` (T24 D4) and `tau2020mobile/` (option B), each with its `classes.json`. They hold no audio and
no features, and they keep the original columns, so they can stand in for the run folders' `csv/` files. Freesound
uploader names are replaced by integers (the `uploader` column of FSD50K, the index `head_name` already carries, and
`u:<index>` in the `group` column of FSDnoisy18k; the same uploader always gets the same integer, so the grouping can be
checked); the FSDKaggle2018, TUT 2017 and TAU lists are byte-identical to the originals.
`data/folds/manifest.csv` gives the MD5 of each original and of each shipped file. The `t21_build_*.py`,
`t24_build_*.py` and `optionB/build_csv.py` scripts regenerate them from the dataset metadata with fixed seeds;
`StratifiedGroupKFold` can assign folds differently across scikit-learn versions, so compare with these lists (the
runs used scikit-learn 1.6.1).

**Quinn.** The library still carries a loader and `config/quinn.yaml` for Quinn, a dataset this study does not use,
because the tests and the result collector read them. Quinn's raw audio is not public and only its embeddings exist,
on the original machine; they are not in this archive. No number in the paper uses Quinn.

## 3. Map from paper results to commands

| Paper result | Code | Run folder (in `$RESULTS_DIR`) | Observed time, M2 Pro |
|---|---|---|---|
| Pre-registered replication, FSDKaggle2018 | `scripts/redesign/moe/t21/` | `2026-10-26_1408_moe-t21_code-731570e` | about 4 h download, 5.7 h extraction to scores; analysis only 2 min |
| Pre-registered replication, FSD50K single-label subset | same | same | about 12 h after download start (11 h of it extraction); analysis only 8 min |
| Main tables, ESC-50 (50 classes), ESC-50 (five groups), ESC-50 (official categories, secondary), UrbanSound8K | `scripts/redesign/idea6/stage3c/` on top of stage 2 and stage 3b | `2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1` | 7 h for stage 3c; the stages it reads take about 2.5 days in all (section 3.2); analysis only 10 to 20 min per task |
| Main tables, TAU 2020 Mobile (device shift) | `scripts/redesign/idea6/optionB/` | `2026-10-23_2133_redesign-idea6-optionB_code-f320295` | 16.5 h including the 27 GB download |
| Humility curve (escalation rate against shift) | stage 3c, stage 2, final round, T21 | see section 3.3 | analysis only, minutes |
| Gate quality by precision | `scripts/redesign/idea6/gatecross/` | `2026-10-24_1557_gate-by-precision_code-6cbc780` | 8 min, CPU, from stored scores |
| Three-tier mixture against the three-model average (T27, post hoc) | `scripts/redesign/moe/t27/` | `2026-10-29_1538_moe-t27_code-856efb5` | analysis only, about 6 min (recompute about 8 min), from stored scores |
| Negative results | section 3.5 | section 3.5 | section 3.5 |

**Tables and figures by label.** Numbers are those of the 28 October build of the paper and may move; the labels do
not. "Run folder" names are under `$RESULTS_DIR`. The scripts that draw the three data figures are in `scripts/figures/`
(its README); each asserts that what it draws matches the tables named here.

| Label (number) | What it shows | Tables it reads | Command |
|---|---|---|---|
| `fig:overview` (Figure 1) | the two-expert cascade | none | schematic; the design is the T21 cascade (3.1) |
| `tab:t21` (Table 1) | fresh-data test: FSDKaggle2018 field and clean, the wider-hull rows, FSD50K, FSDnoisy18k and TUT 2017 | T21 `tables/d1/`, `tables/d2/` (`primary_by_condition.csv`, `verdict.json`); T25 `tables/d1/verdict.json`, `t25_by_condition.csv`; T24 `tables/d3/`, `tables/d4/primary_by_condition.csv`, `tables/verdict_combined.json` | `t21_analyze.py` (3.1), `t25_analyze.py` and `t24_analyze.py` (3.1.1) |
| `fig:t21` (Figure 2) | the cascade against the hull on FSDKaggle2018 and FSD50K; gain on inner validation per budget | T21 `tables/d1/verdict.json`, `tables/d2/verdict.json`; T25 `tables/d1/verdict.json` | as `tab:t21`; figure: `scripts/figures/fig-t21-fresh-data-v2.py` |
| `tab:hull` (Table 2) | development cascades against the equal-cost hull, field mixture; always averaging added to the hull | stage 3c `analysis/<task>/` and its README table "P3 in detail"; option B `analysis/verdict.json`; reviewer extras `A/a_ensemble_hull.csv` | `analyze_3c.py`, `summary_3c.py`, `posthoc_unpadded.py`, `analyze_optb.py` (3.2); `python scripts/redesign/idea6/reviewer_extras/a_ensemble_hull.py --s3c S3C --s3b S3B --s2 S2 --i5 I5 --out A` in `2026-10-24_1818_reviewer-extras_code-ac7e49d` |
| `tab:humility` (Table 3) | escalation rate by noise kind and SNR, 30% budget | T3/T5 `tables/t5_rate_by_cell_q30.csv` | `t5_score.py`, `t5_analyze.py` (3.5, T3 and T5) |
| `fig:humility` (Figure 3) | escalation against SNR, ESC-50, UrbanSound8K, FSDKaggle2018, FSD50K | as `tab:humility`, plus T21 `tables/d1/verdict.json` and `tables/d2/verdict.json` (budgets b20, b30) | as `tab:humility` and `tab:t21`; figure: `scripts/figures/fig-humility.py` |
| `fig:gateprec` (Figure 4), `tab:gateprec` (Table 8) | margin-gate AUROC against FP32 per backbone and head precision | gate by precision `cross_compact.md` ("Cross table (compact)", all conditions) | `gatecross/run_all.sh`, `compact_cross.py` (3.4); figure: `scripts/figures/fig-gate-by-precision-v2.py` |
| `tab:all` (Table 4) | every design tested | the README of the run folder in each row | the row's section: 3.1 (T21, T24, T25), 3.2 (idea 6 stages 2 to 3c, option B, final round), 3.4 (gate by precision), 3.5 (everything else), 3.7 (T27, post hoc, row moe-t27). The audit rows (idea 5 stage 3, idea 6 stage 3c, T19 and T21 audits): `scripts/audits/` (its README). Idea 5 stage 1, idea 6 stage 1 and the head-precision headroom: 3.8 |
| `sec:fresh` (Section 5.1), paragraph "A three-tier mixture, post hoc" | T27: the three-tier mixture (T26's S2) against always averaging MN10-Q8, MN20-Q8 and AST-Q4, and against the widened hull with that average added; FSDnoisy18k and FSDKaggle2018, field and clean | T27 `tables/summary.csv`, `tables/summary.json`, `tables/<d>/results.json` | `t27_analyze.py`, `t27_recompute.py` (3.7) |
| `tab:t21cond` (Table 5) | the fixed cascade on FSDKaggle2018 by test condition | T21 `tables/d1/primary_by_condition.csv`, `gate_auroc_by_condition.csv` | `t21_analyze.py --r $R --d d1` (3.1) |
| `tab:astclean` (Table 6) | clean F1 of the transformer at 1024 frames by weight width and scale layout | idea 5 stage 2 `analysis/{esc50,esc5,urban}/static.csv` | `analyze_all.sh` (3.2, row 1), then `python scripts/redesign/inventory/ast_static_clean.py` |
| `tab:heads` (Table 7) | EfficientNet-B3 heads by bit width against the FP32 head | protocol 3 singles `2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs/<ds>_singles_s<seed>/<kind>/fold_k/metrics.json` | `bash scripts/redesign/singles_v3.sh --list` with `ESC_CSV`, `URBAN_CSV` and `V3_OUTPUTS=$RESULTS_DIR/2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs` (its header gives the rest), run as a job list; then `python scripts/redesign/inventory/heads_table.py`. That script also reads the protocol 2 run (`2026-10-20_1300_protocol-v2_code-5db7e75`), which is not in the paper and has no command here |
| `tab:names` (Table 9) | names and configuration codes | none | no command; the codes are the configuration names used by `QWave/backbone_quant.py` and the stage scripts |
| `tab:latency` (Table 10) | median CPU time per step at 2 threads | latency run `tables/steps.csv` | 3.2.1 |

**Numbers in the text outside the tables.**

| Where | Numbers | Source | Command |
|---|---|---|---|
| `sec:fresh`, paragraphs on FSDKaggle2018, the cost result, the wider hull, FSD50K, FSDnoisy18k and TUT 2017 | gains, intervals, costs, escalation | T21, T24, T25 tables | 3.1, 3.1.1 |
| `sec:fresh`, paragraph "The audit" | 2.830 and 2.070 rebuilt; +2.61 with a fixed AST cost; +2.87 without 187 near duplicates; +2.86 without 57 clips; int8 +4.19 | the two T21 audit folders; T21 `tables/d1/verdict.json` (`secondary_int8`) | `scripts/audits/t21-audit/`, `t21-audit-d2/`; `t21_analyze.py` |
| `sec:fresh`, paragraph "Routing between two small CNNs" | preconditions +2.14 and +0.51; +0.39 over the widened hull; 69.24 at 0.057 T; recompute +0.69 | T26 `tables/c/` | 3.6 |
| `sec:development` | +2.11, +1.85, 0.45 below always averaging, +1.29, +1.03, TAU | stage 3c `analysis/`, reviewer extras `A/` | 3.2; `a_ensemble_hull.py` (`tab:hull` row) |
| `sec:development`, strong expert alone | +1.41, -0.21, -0.07 | stage 3c `analysis/<task>/e2.csv` | `analyze_3c.py` (3.2) |
| `sec:development`, the experts of the ESC-50 pick and the runner-up | 0.857, 0.855; 0.07 points | stage 3c `analysis/esc50/static.csv`, `selected.csv`; stage 3c audit | `analyze_3c.py`; `scripts/audits/idea6-stage3c-audit/c2_p3_recompute.py` |
| `sec:bits`, heads | within 0.6 points; Nadeau-Bengio TOST p 0.0505 to 0.20 (ESC-50) and at most 0.024 (UrbanSound8K); plain TOST p at most 0.043; BitNet head losses, MN10 and MN20 heads within 0.41 | `tab:heads` sources; reviewer extras `B/` | `heads_table.py`; `scripts/audits/fact-checks/table1_fold_level.py`; `scripts/audits/fact-checks/tost_recompute.py $RESULTS_DIR/2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs`; `reviewer_extras/b_heads.py` |
| `sec:bits`, the transformer | 0.3 to 0.5 and 1.8 to 2.3 points at four bits; 1.6 and 11.7 under white noise at 0 dB | idea 5 stage 2 `analysis/`; fact checks | `ast_static_clean.py`; `scripts/audits/fact-checks/` (`train_w4_heads.sh`, `q4ch_head_transfer.py`, `q4ch_summary.py`) |
| `sec:bits`, real int8 and the scaler rule | 30 to 63 and 74 to 94 points lost; one dimension shifted 147 SD; 94 to 98% and 93 to 97% recovered; retrained heads 0.4 to 0.6 below | stage 3c README tables; reviewer extras `C/`; stage 3c audit | 3.2 (stage 3c); `reviewer_extras/c_scaler_curve.py`, `c_summary.py`; `scripts/audits/idea6-stage3c-audit/a1_compare_emb.py`, `a2_scaler_test.py` |
| `sec:humility` | escalation rates and AUROCs on the fresh data; development rates | T21, T24 `tables/<d>/humility.csv`, `gate_auroc_by_condition.csv`; T3/T5 tables | 3.1, 3.1.1, 3.5 (T3 and T5) |
| `sec:stats` | family of 22, p 0.038, variance ratios 0.44 to 1.07 and 0.71 to 0.92 | T21 `verdict.json`; T14 tables | 3.1; `R=R bash scripts/redesign/moe/run_t14.sh` (3.5) |
| `app:setup` | FSD50K absent classes, 2.04 and 2.05, 1.98 without the two parent classes; 19 and 187 near duplicates | the two T21 audit folders | `scripts/audits/t21-audit-d2/`, `t21-audit/` |
| `sec:earlier` (Appendix A) | none in the review version | | |

**Bit-exactness of re-extraction.** On the hardware and software of section 1, every extractor rebuilds the stored
embeddings and score files bit for bit, checked on 64 clips per dataset (seed 0) for all seven datasets. MN10-Q8 on the
CPU is the exception when only part of a clip list is extracted: clips of equal length in a 128-row chunk share a
batch, a different batch changes the last bits of the FP32 activations, and the 8-bit fake quantization can turn that
into whole quantization steps (up to 2.6% of an embedding; 10 of 5,120 argmax decisions differed on the subsets).
Extracting the whole list in list order, as the launchers do, gives bit-identical files. Other hardware was not tested.
Noise draws are seeded per clip (`crc32(name|kind)`), so they do not depend on batch order or list length.

### 3.1 Pre-registered replication on unseen data (T21)

Design fixed in `scripts/redesign/moe/t21/ANALYSIS_PLAN.md` before any audio of either dataset was downloaded: MN10-Q8
(W8A8, CPU, each clip at its own length) answers every clip; when its top-1 minus top-2 margin is below the threshold for
the budget chosen on clean inner validation, AST-Q4 (w4g128, each clip at its own frame count) also runs and the two
probability vectors are averaged. Conditions: clean, background, pink and white noise at 10 and 0 dB, and the field
mixture.

**FSDKaggle2018 (D1).** Five stratified folds over the 5,310 manually verified clips, five seeds.

```bash
bash scripts/download_datasets.sh fsdk2018; bash scripts/download_datasets.sh urban8k   # UrbanSound8K is the background-noise bank
T=scripts/redesign/moe/t21; R=$RESULTS_DIR/2026-10-26_1408_moe-t21_code-731570e; mkdir -p $R/logs
python -I $T/t21_build_d1.py ~/Documents/DATASETS/FSDKaggle2018/meta_x/FSDKaggle2018.meta $R/csv/d1
R=$R D=d1 bash $T/t21_cpu_mn.sh                                  # MN10 FP32 and W8A8, every condition, log-mel cache (CPU)
R=$R D=d1 bash $T/t21_gpu_ast.sh check pool_q pool_f extra        # AST FP32 and w4g128 (MPS)
python $T/t21_make_csvs.py --r $R --d d1
R=$R D=d1 bash $T/t21_train_heads.sh pool official               # FP32 heads, protocol 3 recipe (CPU)
for s in export build embed; do $PYO $T/t21_int8.py $s --r $R --d d1; done   # secondary: real int8 MN10 on ONNX Runtime
python $T/t21_score.py --r $R --d d1
python $T/t21_check_emb.py $R d1
python $T/t21_analyze.py --r $R --d d1                            # tables/d1/: verdict.json, primary_*.csv, humility.csv
```

**FSD50K single-label subset (D2).** 113 classes, five folds grouped by uploader, three seeds.

```bash
bash scripts/download_datasets.sh fsd50k-meta
python -I $T/t21_build_d2.py ~/Documents/DATASETS/FSD50K/meta_x $R/csv/d2
R=$R bash $T/t21_get_d2.sh                                        # audio zips with MD5 check; extracts only the subset's clips
R=$R AUD=~/Documents/DATASETS/FSD50K/audio_raw/dev AUDE=~/Documents/DATASETS/FSD50K/audio_raw/eval D=d2 bash $T/t21_cpu_mn.sh
R=$R AUD=~/Documents/DATASETS/FSD50K/audio_raw/dev AUDE=~/Documents/DATASETS/FSD50K/audio_raw/eval D=d2 bash $T/t21_gpu_ast.sh pool_q pool_f extra
R=$R bash $T/t21_d2_chain2.sh mn & R=$R bash $T/t21_d2_chain2.sh ast; wait   # CSVs, heads, scores, analysis
python $T/t21_analyze.py --r $R --combine d1 d2                   # tables/verdict_combined.json (Holm over datasets)
```

The FSD50K log-mel cache reaches 22 GB, and `t21_d2_chain2.sh mn` deletes it while it builds the int8 embeddings. In
the original run a disk guard (`t21_disk_guard.sh`) deleted the cache before the int8 step, so the FSD50K int8
secondary was not run; the README of the run folder lists this and the other deviations.

#### 3.1.1 Two more fresh datasets (T24) and the wider hull (T25)

T24 runs the T21 cascade unchanged on FSDnoisy18k (D3) and TUT Acoustic Scenes 2017 (D4), with fixed official test
sets (`cross_validation.fixed_test=true` in `scripts/benchmark.py`). Both datasets failed the precondition, so the
rows in the paper are descriptive. Run folder `2026-10-28_1827_moe-t24_code-9a51cf5`. With `T=scripts/redesign/moe/t24`:

```bash
R=$R bash $T/t24_get_d3.sh meta; python -I $T/t24_build_d3.py <FSDnoisy18k meta> <FSD50K meta_x> $R/csv/d3; R=$R bash $T/t24_get_d3.sh audio
R=$R D=d3 AUD=<prepared audio> bash $T/t24_cpu_mn.sh all_f all_q eval_q; R=$R D=d3 AUD=<prepared audio> bash $T/t24_gpu_ast.sh all_f eval_q all_q
python $T/t24_make_csvs.py --r $R --d d3; R=$R D=d3 SEEDS="0 1 2" bash $T/t24_train_heads.sh
python $T/t24_check_emb.py $R d3; python $T/t24_score.py --r $R --d d3 --seeds 0 1 2; python $T/t24_overlap.py --r $R --d d3
python $T/t24_analyze.py --r $R --d d3 --seeds 0 1 2 --drop $R/tables/d3/overlap_drop.txt; python $T/t24_recompute.py --r $R --d d3 --seeds 0 1 2
# D4: the same with t24_get_d4.sh and t24_build_d4.py <dev meta> <eval meta>, seeds 0 to 4, no --drop
python $T/t24_analyze.py --r $R --combine d3 d4
```

`t24_overlap_variant.py` is a descriptive variant written after the D3 scores and is not pre-registered.

T25 scores the same T21 cascade against a hull widened with MN04, MN05, MN20 and DyMN04 at simulated 8 bits and every
average of two of the five CNNs, on FSDKaggle2018. It reads T21's stored scores. Run folder
`2026-10-28_2247_moe-t25_code-cc3b048`, about 4 h of CPU extraction:

```bash
R=$R bash scripts/redesign/moe/t25/t25_cpu_chain.sh              # extraction (CPU), checks, CSVs, heads, scores
python scripts/redesign/moe/t25/t25_analyze.py --r $R --d d1; python scripts/redesign/moe/t25/t25_analyze.py --r $R --combine d1
python scripts/redesign/moe/t25/t25_recompute.py --r $R --d d1     # separate recompute from the score files
```

### 3.2 Main tables: ESC-50, UrbanSound8K (stage 3c) and TAU 2020 Mobile

The ESC-50 and UrbanSound8K tables are built in stages. Each stage reads the stored outputs of the stages before it,
so run them in this order. Every launcher under a `run/` folder is the script the original run used, with machine
paths replaced by the variables of section 1.

| Order | Stage | Run folder | Launchers, in order | Observed time |
|---|---|---|---|---|
| 1 | AST at 1024 frames, weights W8 to W2, clean and noise (idea 5 stage 2) | `2026-10-20_2359_redesign-idea5-stage2_code-3c18c38` | `scripts/redesign/idea5/run/`: `extract_esc50_clean.sh`, `extract_noisy.sh`, `extract_esc50_g64.sh`, `extract_urban_w6.sh`, `extract_esc50_20db.sh`, `train_heads.sh`, `train_heads_s34.sh`, `train_cheaphead.sh`, `analyze_all.sh esc50\|urban\|esc5` | 36 h (MPS extraction, with pauses); analysis 10 min |
| 2 | Audio statistics for the model-free baselines (idea 7 stage 1) | `2026-10-22_1105_redesign-idea7-stage1_code-e2af723` | `bash scripts/redesign/idea7/run_all.sh <run folder>` | 1.5 h |
| 3 | EfficientNet-B3 tier, AST at real clip length, cross-dataset and unseen-class shift, latency (idea 6 stage 2) | `2026-10-22_1120_redesign-idea6-stage2_code-9f346ba` | `scripts/redesign/idea6/stage2/run/`: `extract_all.sh`, `extract_ast_short.sh`, `extract_ast_short_w4.sh`; then `stage2/make_csv.py`, `train_heads.sh 4`, `train_ast_short_heads.sh 3`, `train_esc5off.sh 2`, `realshift.py prepare`, `train_realshift.sh 6`, `realshift.py e6`, `latency.py --out $R/latency`; `run/final_run.sh` | 5 h; `final_run.sh` 13 min |
| 4 | EfficientAT MN04 and MN10, simulated W8A8 (stage 3b) | `2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db` | `scripts/redesign/idea6/stage3b/run/`: `extract_ast.sh`, `extract_cnn.sh`, `extract_cnn_resume.sh`, `extract_cnn_urban_noisy.sh`; `R=$R bash stage3b/make_csvs.sh`; `R=$R MAXRUN=6 bash stage3b/train_heads.sh 6`; `stage3b/check_embeddings.py $R`; `run/final_run.sh`, `run/final_analyze.sh` | 19 h |
| 5 | MN05, MN20, DyMN04; real int8 on ONNX Runtime; exact-length UrbanSound8K (stage 3c) | `2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1` | below | 7 h |

Stage 3c, with `R` the stage 3c folder, `S3B`, `S2`, `I5` and `I7` the folders of rows 4, 3, 1 and 2, and `WT` the EfficientAT weights:

```bash
H=scripts/redesign/idea6/stage3c
bash $H/run/extract_3c.sh                                  # MN05, MN20, DyMN04 on both datasets; MN10 at exact length (MPS)
R=$R bash $H/make_csvs.sh; R=$R bash $H/train_heads.sh 4
bash $H/run/mels.sh esc50; bash $H/run/mels.sh urban8k     # float16 log-mel caches for ONNX Runtime
bash $H/run/ort_prepare.sh mn10 mn04; bash $H/run/after_esc50.sh; bash $H/run/ort_urban_early.sh
bash $H/run/ort_mn20.sh; bash $H/run/int8b_mn20_more.sh
R=$R bash $H/train_int8_heads.sh 3 < $R/logs/int8b_jobs.txt; bash $H/run/after_int8b.sh
R=$R S3B=$S3B bash $H/score_all.sh unp cnn
bash $H/run/final_run.sh                                   # analyze_3c.py for esc50, esc5, urban, esc5off; summary_3c.py
python $H/posthoc_unpadded.py --r $R --s3b $S3B            # exact-length heads (reported as post hoc)
```

Per task, `analyze_3c.py --task esc50|esc5|esc5off|urban` writes `analysis/<task>/` (static front, hull, picks,
`s2_humility.csv`, `verdict.json`), and `summary_3c.py` writes `analysis/summary.md`. The MN10-Q8 to AST-Q4 pair is one
of the designs these files score on every task.

**TAU 2020 Mobile (option B).** Device A is the training and clean condition; devices B, C and S1 to S6 are the shift.

```bash
OB=scripts/redesign/idea6/optionB; R=$RESULTS_DIR/2026-10-23_2133_redesign-idea6-optionB_code-f320295
D=~/Documents/DATASETS/TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development
META=~/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv
bash $OB/download_tau2020.sh
python $OB/build_csv.py --meta $D/meta.csv --eval-setup $D/evaluation_setup --out $META
R=$R bash $OB/queue_mps.sh                                  # extract_optb.sh cnn, then ast (MPS)
python $OB/audio_stats.py --csv $META --audio-dir $D/audio --bg-csv ~/Documents/DATASETS/ESC-50/ast_768/esc-50.csv \
  --bg-audio-dir ~/Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio --conditions clean --out $R/audio_stats/tau
R=$R bash $OB/pipeline_optb.sh 2                            # training CSVs, 30 heads, scores
python $OB/ort_mels_optb.py --csv $META --audio-dir $D/audio --out $R/cache_mel/tau --conditions clean
for m in mn20 mn10; do for s in export build embed; do $PYO $OB/ort_int8_optb.py $s --model $m --r $R; done; done
python $OB/analyze_optb.py --r $R --meta $META
python $OB/score_int8_optb.py --model mn20 --kinds 8 --r $R --meta $META; python $OB/analyze_int8_optb.py --r $R --meta $META --model mn20
python $OB/summary_optb.py --r $R
```

**Final round (non-additive corruptions and real shift on the small CNNs).** Run folder
`2026-10-24_1507_redesign-idea6-final_code-ed82400`, 2 h. In `scripts/redesign/idea6/final/`, with `R` set to that
folder (`env.sh` sets the input folders): `check_extract.sh`, `extract_final.sh 1`, `extract_final.sh 2`,
`realshift_final.py prepare`, `extract_partA.sh`, `run_audio_stats.sh`, `score_final.sh esc`, `score_final.sh urban`,
`train_xds.sh` (runs `realshift_final.py build-csv`), `run_analysis.sh B` (`analyze_corrupt.py` per task, `check_repro.py`),
`run_analysis.sh A` (`realshift_final.py e6`, `e7`), then `summary_final.py` and `summary_a.py`.

#### 3.2.1 CPU latency (Table 10)

Run folder `2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6`, about 3.5 h. The timing scripts run from a copy
inside the run folder, as in the original run; `W` is this repository.

```bash
R=$RESULTS_DIR/2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6; mkdir -p $R/scripts $R/logs
cp scripts/redesign/idea6/latency_full/* $R/scripts/; cd $R; export W=<repository root> PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 $PY scripts/prep_inputs.py
bash scripts/run_timing.sh 2 1 4; bash scripts/rerun_noisy.sh     # torch and ONNX Runtime blocks per thread count
$PY scripts/cascade_latency.py; $PY scripts/make_tables.py         # tables/steps.csv, cascade_latency.csv, tables.md
```

### 3.3 Humility curve

The escalation rate of the cheap expert, with one threshold set on clean inner validation, as the audio moves away from
the training conditions. Every number is in a table that the steps above write; no extra run is needed.

| Panel | Table | Written by |
|---|---|---|
| Noise, ESC-50 (50 classes and five groups) and UrbanSound8K, clean to 0 dB, and the field mixture | `analysis/<task>/s2_humility.csv` of stage 3c | `analyze_3c.py` (3.2) |
| Unseen classes and other-dataset clips | `analysis/e7_ood.csv`, `e7_tests.csv`, `e6_cross_dataset.csv`, `e6_tests.csv` of idea 6 stage 2 | `stage2/run/final_run.sh`, `realshift.py e6` (3.2, row 3) |
| Reverberation, clipping, low-pass filtering; real shift on MN10 and MN20 | `analysis/` of the final round (`escalation_summary.csv`, `summary_b.md`, `summary_a.md`) | `final/run_analysis.sh` (3.2) |
| Device shift, TAU | option B, `analysis/` (S2 humility) | `analyze_optb.py` (3.2) |
| FSDKaggle2018 and FSD50K, clean to 0 dB and field mixture, with gate AUROC per condition | `tables/<d>/humility.csv`, `gate_auroc_by_condition.csv` of T21 | `t21_analyze.py` (3.1) |
| Escalation rate by noise kind and SNR at the 30% budget (Table 3), ESC-50 and UrbanSound8K | `tables/t5_rate_by_cell_q30.csv` of T3/T5 | `t5_score.py`, `t5_analyze.py` (3.5) |

### 3.4 Gate quality by precision

Margin-gate AUROC for "the model is wrong", calibration error and escalation, compared with FP32 at every backbone and
head precision that the stages above stored. CPU only, about 8 minutes:

```bash
OUT=$RESULTS_DIR/2026-10-24_1557_gate-by-precision_code-6cbc780 bash scripts/redesign/idea6/gatecross/run_all.sh
python scripts/redesign/idea6/gatecross/compact_cross.py --out $OUT
```

It reads the stored scores of idea 5 stage 2, idea 6 stage 2, stage 3b, stage 3c (padded and exact length), option B
and option C. `crosscheck_f1.py` checks that the macro F1 of 157 systems equals the earlier `static.csv` values.

### 3.5 Negative results

Each test was pre-registered in its plan, with the success bar and the stop rule fixed before any test score. Drivers
and analysis scripts take their arguments as their docstrings state.

| Test | What failed | Plan and code | Run folder | Observed time |
|---|---|---|---|---|
| Learned router inside the QMoE head (idea 4) | a router trained on expected loss, with an error head and hard escalation, does not beat the margin | `scripts/redesign/idea4/` (snapshot of the modified library, see its README) | `2026-10-20_2039_redesign-idea4-stage1_code-5ecd753` | 1.3 h |
| Cascade on head precisions (idea 1) | escalating from a low-bit head to a higher-bit one loses to time-sharing | `scripts/redesign/idea1/` | `2026-10-20_2041_redesign-idea1-stage1_code-7f82477`, `2026-10-21_0002_redesign-idea1-stage2_code-85e912a` | 2.2 h, 5 h |
| Learned deferral router (idea 2) | cross-fitted router on the embedding, no gain over the margin | `scripts/redesign/idea2/` | `2026-10-20_2030_redesign-idea2-stage1_code-d702467` | 3 h |
| Disagreement between precisions (idea 3) | adds nothing beyond the margin | `scripts/redesign/idea3/` | `2026-10-20_2041_redesign-idea3-stage1_code-315521e` | 2.1 h |
| Time-sharing hull on the AST precision cascades (idea 5b) | AST precision cascades do not beat the hull | `scripts/redesign/idea5b/run_all.sh` | `2026-10-22_1042_redesign-idea5b-hull_code-fdb2143` | 2 min |
| Model-free deferral on audio statistics (idea 7) | audio statistics do not beat the margin | `scripts/redesign/idea7/` | `2026-10-22_1105_redesign-idea7-stage1_code-e2af723` | 1.5 h |
| Own heads on 3-bit AST tiers, three-tier cascade (option C) | third tier gains 0.85 points at 2.3 times the cost; no pass | `scripts/redesign/idea6/optionC/` (`extract_c.sh`, `train_own_heads.sh 3`, `run_analysis.sh`) | `2026-10-23_2133_redesign-idea6-optionC_code-d6e7913` | 2.2 h |
| Specialization by difficulty: pilot, T1, T1a, T1b, T1c | experts that share one input do not specialize by difficulty | `scripts/redesign/moe/ANALYSIS_PLAN.md`, `ANALYSIS_PLAN_T1B.md`, `ANALYSIS_PLAN_T1C.md`; `s0_t0.py` to `s7_tau_analysis.py`, `run_t1*.py`, `t1b_*.py`, `t1c_*.py` | `2026-10-24_2013_moe-difficulty-pilot_code-b77b98e`, `2026-10-24_2208_moe-t1b_code-4e3e036`, `2026-10-25_0052_moe-t1c_code-7d7429e` | 1.6 h, 7 min, 42 min |
| Learned gate over stored experts (T2b) | does not beat averaging | `ANALYSIS_PLAN_T2B.md`; `t2b_oof.py`, `t2b_fit.py`, `t2b_score.py` | `2026-10-24_2224_moe-t2b_code-f5e4935` | 4 min after the out-of-fold logits |
| Masked-view AST expert (T7) | token masking loses more accuracy than it saves | `ANALYSIS_PLAN_T7.md`; `t7_*.py` | `2026-10-24_2313_moe-t7_code-9fb5b35` | 1.5 h |
| Mixture of diverse stored experts (T13) | gains under shift do not survive the correction | `ANALYSIS_PLAN_T13.md`; `t13_*.py` | `2026-10-25_0145_moe-t13_code-eaaedc9` | 4 min |
| Diversity, distillation, all-clips adapter expert (T17) | no pass on any part | `ANALYSIS_PLAN_T17.md`; `t17_*.py`, `t17c_*.py`, `t17_run/` | `2026-10-25_1415_moe-t17_code-8f33aaf` | 10 h |
| Precision cascade with low-bit activations (T19) | fails on all three tasks; stop rule fires | `ANALYSIS_PLAN_T19.md`; `t19_*.py`, `t19_run/`; `QWave/backbone_quant.py` (activation mode), `QWave/costs.py` | `2026-10-26_1409_moe-t19_code-ae334eb` | 38 h |
| Fine-tuned strong expert (T20) | stops at its precondition | `ANALYSIS_PLAN_T20.md`; `t20_*.py`, `t20_run_*.sh` | `2026-10-26_1408_moe-t20_code-ca935d8` | 20 h |
| Learned router against the margin on stored arrays (T3) and risk-coverage of the margin (T5) | margin ahead of the router; T5 fails one monotonicity clause | `ANALYSIS_PLAN_T3T5.md`; `t3t5_check_inputs.py`, `t3t5_preselect.py`, `t5_score.py`, `t5_ood.py`, `t3_score.py`, `t5_analyze.py --run R`, `t3_analyze.py --run R`, `t3t5_manifest.py --run R` | `2026-10-25_1406_moe-t3t5_code-81a3d6a` | 1.8 h (T5 in minutes, then the T3 router fits) |
| Any-precision tier selection (T9) | fails the +1.0 rule | `scripts/redesign/anyprecision/` (`anyprecision.py --phase select`, then the test phase, per task; `summary.py R`) | `2026-10-24_2014_moe-anyprecision-check_code-0079436` | 10 min |
| Adapter specialist with its own capacity (T12) | stops at its line | `ANALYSIS_PLAN_T12.md` (`ANALYSIS_PLAN_T12_FULL_DRAFT.md` is the full run it would have led to, not run); `t12_cache.py`, `t12_cost.py`, `t12_cost_table.py`, `t12_pilot.py`, `t12_score.py`, `t12_ptq.py` | `2026-10-25_0312_moe-t12-feasibility_code-e469a35` | about 1 h (token cache about 9 min) |
| Stability of every system (T14) | no notion holds | `ANALYSIS_PLAN_T14.md`; `R=R bash scripts/redesign/moe/run_t14.sh` | `2026-10-25_1334_moe-t14-stability_code-e49a488` | 1.3 h |
| Input-view experts (T15) and a frozen coarse tier (T18) | stop rule (T15); coarse tiers too weak (T18) | `ANALYSIS_PLAN_T15.md`, `ANALYSIS_PLAN_T18.md`; `t15_extract.py`, `t15_heads.sh`, `t15_precond_heads.py`, `t15_wave.py`, `t15_survivors.py`, `t15_preselect.py`, `t15_score.py`, `t15_analyze.py`, `t15_costs.py`, `t15_latency.py`; `QWave/panns_cnn14.py`. Results folder path in `~/Documents/qmoe-work/t15-R.txt` or `T15_R` | `2026-10-25_1415_moe-t15-t18_code-2ef25b0` | 6.5 h |
| Router on transformer hidden states (T16) | stopped | `ANALYSIS_PLAN_T16.md`; `t16_gpu_driver.sh` (extraction through `lockrun.sh`), `t16_heads.py`, `t16_router.py`, `t16_score.py`, `t16_analyze.py --score DIR --out DIR`, `t16_cost.py`, `t16_latency.py`; working folder `~/Documents/qmoe-work/t16-data` | `2026-10-25_2006_moe-t16_code-0254307` | about 6 h |
| Scale hierarchy with a trained low-resolution expert (T22) | stops at its precondition | `t22/ANALYSIS_PLAN.md`; `t22/` (`t22_costs.py`, `t22_cache.py`, `t22_p1_driver.sh`, `t22_p1.py`) | `2026-10-28_0117_moe-t22_code-b33df0d` | 4 h |
| Waveform expert, PANNs Wavegram-CNN14 at W8A8 (T23) | fails; stop rule fires | `t23/ANALYSIS_PLAN.md`; `t23/` (`t23_p1_chain.sh`, `t23_ft_all.sh`, `t23_gpu_full.sh`, `t23_score.sh`, `t23_analyze.py`, `t23_recompute.py`); `QWave/panns_wavegram.py` | `2026-10-28_0119_moe-t23_code-4d6bde8` | 14 h |
| MN10-Q8 then MN20-Q8 against the widened hull (T26) | H1-C fails on FSDnoisy18k; TUT 2017 stops at the precondition | section 3.6 | `2026-10-29_0912_moe-t26_code-6ecea17` | about 6 h |

T19 runs `t19_costs.py`, `t19_calib.py` per task, `t19_repro_a8.py`, `t19_run/run_p1_extract.sh`, `run_p1_heads.sh`,
`t19_p1.py`, `t19_run/run_jobs.sh jobs_full.txt`, `run_legacy_heads.sh`, `run_heads.sh <task>`, `after_gpu*.sh`,
`t19_check_emb.py`, `t19_analyze.py --cheap n4a4`. T20 runs `t20_first_turn.sh`, `t20_run_task.sh <task>`, then
`t20_analyze.py --run R --phase p1 --tasks urban esc50` and `t20_descriptive.py --run R --tasks urban esc50`. T23 reads
its results folder from `~/Documents/qmoe-work/t23-R.txt`.

Inputs and paths for these reruns, all checked by rerunning from the stored score files:
- T19 reads its results folder from `T19_R` or, if unset, a `.results_dir` file at the archive root; the T19 audit
  writes next to its scripts unless `AUDIT_OUT` is set.
- T15: `t15_precond_heads.py --r R --views v1 v2 v3 v4 v6 lra lrb --datasets urban esc50 --seeds 0 1 2 --outs
  outputs_precond_fold1` for the stored run (this view order gives byte-identical tables; another order changes only the row order).
- T22 caches in `T22_CACHE` (default `~/Documents/qmoe-work/t22-cache`); `t22_p1.py` needs `t22_cache.py` first, and
  the printed numbers come from `t22_p1_recompute.py`.
- T23: the head-training lists `csv_ft/<task>_f1_s<seed>.csv` are rebuilt from `ft/*/wg_fp32__clean.npy` by lines 11
  to 17 of `t23/t23_p1_ft.sh`.
- Idea 4: `idea4/summarize.py <diag json> [control]` writes the stage 1 verdict.
- `fact-checks/q4ch_head_transfer.py` and `q4ch_auroc.py` read `fact-checks/heads_w4/`; build it first with
  `R=<idea 5 stage 2 run folder> bash scripts/audits/fact-checks/train_w4_heads.sh`.
- `t21-audit-d2/audit_leakage_d2.py` needs the AudioSet ontology: fetch
  `https://raw.githubusercontent.com/audioset/ontology/master/ontology.json` into its `ontology_download/` (SHA-256
  `9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d`).
- Assembled by hand, no script: the final round `summary.md` (`summary_b.md` followed by `summary_a.md`), idea 5b
  `outputs/cost_check.json` and the T15 `tables/*_preselected.json` (copies of computed values); the T1c README cites
  `t2b/descriptive_vs_fixed_E2a.csv`, which has no script and no printed number.

### 3.6 MN10-Q8 then MN20-Q8 against the widened hull (T26)

T26 runs the T21 design with MN20-Q8 in place of AST-Q4 (S1), and a three-tier version that adds AST-Q4 after the
pair (S2), against T25's widened hull. Part A (FSDKaggle2018, stored T21 and T25 scores) is post hoc, Part B
(ESC-50, UrbanSound8K, TAU, stored stage 3c and T5 scores) is descriptive, and Part C (FSDnoisy18k and TUT 2017, T24's
folds and heads recipe) is the pre-registered test. FSDnoisy18k passed the precondition and failed H1-C; TUT 2017
failed the precondition and stopped. The paper reports T26 in Section 5.1 and in the `tab:all` row moe-t26. Plan
`scripts/redesign/moe/t26/ANALYSIS_PLAN.md`, results summary `scripts/redesign/moe/t26/README_results.md`. Run folder
`2026-10-29_0912_moe-t26_code-6ecea17`, about 6 h on the CPU (Parts A and B take minutes from stored scores; Part C
reads T24's run folder for its CSVs and MN10-Q8 and AST-Q4 scores). With `T=scripts/redesign/moe/t26`:

```bash
python $T/t26_part_a.py --r $R; python $T/t26_recompute.py --r $R --part a
python $T/t26_part_b.py --r $R
R=$R D=d3 SEEDS="0 1 2" MODELS=mn20 TAG=pre bash $T/t26_cpu_chain.sh       # extraction, checks, heads, scores, precondition
R=$R D=d4 SEEDS="0 1 2 3 4" MODELS=mn20 TAG=pre bash $T/t26_cpu_chain.sh
R=$R D=d3 SEEDS="0 1 2" MODELS="mn04 mn05 dymn04" TAG=wide bash $T/t26_cpu_chain.sh
python $T/t26_part_c.py --r $R --d d3 --seeds 0 1 2; python $T/t26_recompute.py --r $R --part c --d d3 --seeds 0 1 2
python $T/t26_part_c.py --r $R --d d4 --seeds 0 1 2 3 4 --partial; python $T/t26_part_c.py --r $R --combine d3 d4
python $T/t26_manifest.py $R 6ecea17
```

`t26_recompute_fixed_budget.py $R d3 0.2 0 1 2` was written after the verdict: it reruns the recompute with S1's
budget forced to the analysis's 20%, as a check.

### 3.7 Three-tier mixture against the three-model average (T27, post hoc)

T27 was chosen after T26's results were seen and has no pass bar; every number is descriptive. It keeps T26's S2
unchanged (budgets read from T26's verdict files, thresholds rebuilt on clean inner validation) and compares it with
(i) always averaging MN10-Q8, MN20-Q8 and AST-Q4 on every clip and (ii) T25's widened hull with that average added as
an 18th point, at S2's cost, on FSDnoisy18k (seeds 0 to 2) and FSDKaggle2018 (seeds 0 to 4), field and clean. No
extraction and no GPU: it reads the stored scores of T21, T24, T25 and T26 and T26's run folder
`2026-10-29_0912_moe-t26_code-6ecea17` (verdicts and per-fold tables). Before any new number it rebuilds every T26
row and stops if any differs. Plan `scripts/redesign/moe/t27/ANALYSIS_PLAN.md`, results summary
`scripts/redesign/moe/t27/README_results.md`. Run folder `2026-10-29_1538_moe-t27_code-856efb5`, about 6 min for the
analysis and 8 min for the recompute on the CPU. With `T=scripts/redesign/moe/t27`:

```bash
python $T/t27_analyze.py --r $R
python $T/t27_recompute.py --r $R       # own primitives from t26_recompute.py, raw scores only
python $T/t27_manifest.py $R 856efb5
```

### 3.8 Idea 5 stage 1, idea 6 stage 1 and the head-precision headroom

These three rows of `tab:all` come from launchers kept in their run folders; the scripts they call are in this archive.

```bash
# Idea 5 stage 1 (2026-10-20_2035_redesign-idea5-stage1_code-7bb31c3): AST embeddings per weight width and condition (MPS);
# the same for UrbanSound8K with ESC-50 as the background bank
C="--csv ~/Documents/DATASETS/ESC-50/ast_768/esc-50.csv --audio-dir <ESC-50 audio> --bg-csv ~/Documents/DATASETS/Urban8k/ast_768/urbansound8k.csv --bg-audio-dir <UrbanSound8K audio> --out-dir $R/emb/esc50"
python scripts/redesign/idea5/extract_conditions.py $C --conditions fp32:clean:- fp32:bg:10 fp32:bg:0 w8:clean:- w8:bg:10 w8:bg:0 \
  w4:clean:- w4:bg:10 w4:bg:0 w4a8:clean:- w4a8:bg:10 w4a8:bg:0 w3:clean:- w2:clean:-
python scripts/redesign/idea5/extract_conditions.py $C --conditions fp32:white:20 fp32:white:10 fp32:white:5 fp32:white:0 \
  fp32:pink:20 fp32:pink:10 fp32:pink:5 fp32:pink:0 fp32:bg:20 fp32:bg:5
# heads: scripts/benchmark.py (protocol 2, seed 0, official folds) on the clean FP32 AST CSVs, tags astesc5_a_s0 (bitnet, 1, 4),
# astesc5_b_s0 (8, esc) on class_id, astesc50_s0 (esc, 4) on label, asturban_esc_s0 and asturban_4_s0; then
python scripts/redesign/idea5/score_heads.py --runs $R/outputs/<tag> ... --emb-dir $R/emb/esc50 --out $R/scores/esc50
python scripts/redesign/idea5/analyze.py --scores $R/scores/esc50 --emb-dir $R/emb/esc50 --runs-root $R/outputs \
  --task esc5=astesc5_a_s0,astesc5_b_s0 --task esc50=astesc50_s0          # analysis/esc50/analysis.json, tables.md

# Idea 6 stage 1 (2026-10-22_0953_redesign-idea6-stage1_code-ec70260); S2 = idea 5 stage 2 folder, P3 = protocol 3 singles outputs
R=$R bash scripts/redesign/idea6/train_eff50.sh 3
R=$R P3=$P3 bash scripts/redesign/idea6/score_all.sh esc5 urban esc50
python scripts/redesign/idea6/costs_idea6.py --ast-emb $S2/emb/esc50 $S2/emb/urban8k --out $R/costs.json
for t in esc50 esc5 urban; do python scripts/redesign/idea6/analyze_idea6.py --task $t --eff-scores $R/scores/$t \
  --ast-scores $S2/scores/$t --costs $R/costs.json --out $R/analysis/$t; done
python scripts/redesign/idea6/summary_idea6.py $R/analysis --out $R/analysis/summary.md

# Head-precision headroom (2026-10-20_2300_redesign-headroom2-esc50class-tinyhead_code-bc4ff1c): one protocol 3 job per
# head kind (esc, 1, 2, 4, 8, bitnet) on the 50-class EfficientNet-B3 CSV, then the summary
python scripts/benchmark.py --config-path config --config-name esc50 experiment.datasets.esc.csv=<ESC-50 EfficientNet-B3 CSV> \
  experiment.datasets.esc.label_col=label experiment.datasets.esc.normalization_type=standard experiment.device=cpu \
  experiment.cross_validation.inner_split_groups=recording "experiment.models_to_run=['4']" experiment.seed=0 \
  experiment.output_root=$R/outputs_c50 hydra.run.dir=$R/outputs_c50/esc_singles_s0
python scripts/redesign/inventory/c50_heads.py
```

## 4. Tests

```bash
PYTHONPATH=. pytest tests/
```

`tests/test_reproduce_script.py` and `tests/test_singles_v3.py` call `python` through the shell, so the environment
must be active (or its `bin/` on `PATH`). On a machine with an Apple GPU, `tests/test_backbone_quant.py` also runs its
MPS cases; elsewhere they are skipped.

## 5. Analysis plans and their timing

The plans are the `ANALYSIS_PLAN*.md` files next to the code. The table gives, for each plan, the commit that added it
and each later commit that changed it (git log on the experiment's branch), against the first output that holds a
score of that experiment. All times are UTC on 2026-10-22 to 2026-10-29 (Irish time was UTC+1 until 25 October).
The archive copies of the plans differ from their committed versions only in wording that points to private
material: the source of each round's plan reads "the round's plan (not distributed)", references to the project's own
notes and log read "the project notes" and "the project log", and the T21 to T26 plans lose one table row that listed
the documents an agent could not edit. No rule, threshold or number changes.

How the columns were filled. Plan and amendment times are author dates, the time a commit was first made; two branches
were rebased later, which changed only their committer dates (noted in the row). The first scored output is the
earliest modification time of a score file in the run folder: per-fold `metrics.json` of trained heads, logits and
probabilities (`.npz`, `.npy`), and score or result tables (`.csv`, `.json`). Configurations, splits, input checks,
selections on inner validation, precondition folders, costs, timing, features, logs and copied scripts are left out.
Where the run folder was filled by a copy, or the run kept its own record, the row says which source was used.
A star marks an amendment made after that first score; each such amendment states its own scope, given in the notes.

| Experiment | Plan | Plan commit | Amendments | First scored output | Notes |
|---|---|---|---|---|---|
| Idea 6 stage 2 | `idea6/stage2/ANALYSIS_PLAN.md` | `14c8504` 22 Oct 10:26 | `ed8b742` 22 Oct 10:41, `fa462ce` 22 Oct 11:00, `51dcb2a` 22 Oct 11:05* | 22 Oct 11:03, first noisy test fold, `logs/analysis_plan_commit.txt`; clean ESC-50 head folds were scored from 22 Oct 10:23, before the plan, as its header says | amendment 3 (official categories) came after noisy scoring started (named in the paper) |
| Idea 6 stage 3b | `idea6/stage3b/ANALYSIS_PLAN.md` | `5280b3b` 22 Oct 16:04 | `6bd2bec` 23 Oct 07:30*, `8a8b2aa` 23 Oct 07:30* | 22 Oct 16:08, `outputs/mn10_esc50_s0/esc/fold_1/metrics.json` | amendment 1 came before any UrbanSound8K CNN head |
| Idea 6 stage 3c | `idea6/stage3c/ANALYSIS_PLAN.md` | `47b0cb7` 23 Oct 11:32 | none | 23 Oct 11:35, `outputs/mn05_esc50_s0/esc/fold_1/metrics.json` |  |
| Option B (TAU) | `idea6/optionB/ANALYSIS_PLAN.md` | `4468dad` 23 Oct 20:38 | none | 23 Oct 22:01, `outputs/mn10_tau_s0/esc/fold_1/metrics.json` |  |
| Option C | `idea6/optionC/ANALYSIS_PLAN.md` | `dfb54d5` 23 Oct 20:36 | none | 23 Oct 20:39, `outputs/own_w3g64_urban_s1/esc/fold_1/metrics.json` |  |
| Final round | `idea6/final/ANALYSIS_PLAN.md` | `2a05e77` 24 Oct 14:06 | none | 24 Oct 14:19, `scores/esc50/mn20_esc50_s0/esc/mn20_w8a8__clip_p10.npz` |  |
| Pilot (T0, T1a, T1) | `moe/ANALYSIS_PLAN.md` | `b77b98e` 24 Oct 19:13 | `9ed9d77` 24 Oct 19:30* | 24 Oct 19:21, `t1/raw/s0_f1.npz` | rebased: committed again at 19:22; the run README names the original commit `ec5929d`; amendment 1 (TAU) came after the UrbanSound8K scores; the TAU outputs were opened after the UrbanSound8K verdicts |
| T9 | `anyprecision/ANALYSIS_PLAN.md` | `06e84d1` 24 Oct 19:11 | none | 24 Oct 19:17, `analysis/esc50/tier_diagnostics.csv` | rebased: committed again at 19:23; the run README names the original commit `d434f30`; the selection on inner validation ran from 19:14 |
| T1b | `moe/ANALYSIS_PLAN_T1B.md` | `6d0a653` 24 Oct 21:07 | none | 24 Oct 21:12, `urban/raw_strength/s0_f1.npz` |  |
| T2b | `moe/ANALYSIS_PLAN_T2B.md` | `a39833b` 24 Oct 21:24 | none | 24 Oct 21:28, `urban/raw/s0_f1.npz` |  |
| T7 | `moe/ANALYSIS_PLAN_T7.md` | `b0bf82f` 24 Oct 22:12 | `418ed0d` 24 Oct 22:14, `9fb5b35` 24 Oct 22:16, `ebd9342` 24 Oct 22:35* | 24 Oct 22:35, `raw/s0_f1.npz` | amendment 2 (40 s after that score) covers ESC-50 only, before any ESC-50 head; the first score was UrbanSound8K |
| T1c | `moe/ANALYSIS_PLAN_T1C.md` | `3555e19` 24 Oct 23:48 | `da3cb47` 24 Oct 23:49 | 24 Oct 23:52, `esc50/raw_strength/s0_f1.npz` |  |
| T13 | `moe/ANALYSIS_PLAN_T13.md` | `3875d09` 25 Oct 01:43 | none | 25 Oct 01:49, `tables/all_combos_pooled_shift.csv` |  |
| T12 | `moe/ANALYSIS_PLAN_T12.md` | `1a07674` 25 Oct 02:17 | `47f9cfa` 25 Oct 02:53 | not recoverable: every file of the run folder carries 25 Oct 03:11, the time it was copied; the plan commit says it came before any training run |  |
| T14 | `moe/ANALYSIS_PLAN_T14.md` | `987a4f6` 25 Oct 13:30 | `b06b6a7` 25 Oct 13:34, `5397b1c` 25 Oct 13:56* | 25 Oct 13:35, `tables/plan_v1_cleaniv/primary_sampling.csv` | amendment 2 changed the comparator after a first run (named in the paper) |
| T3, T5 | `moe/ANALYSIS_PLAN_T3T5.md` | `5c960f6` 25 Oct 14:01 | none | 25 Oct 15:02, `tables/t3_mean_auroc.csv` |  |
| T15 | `moe/ANALYSIS_PLAN_T15.md` | `1d37676` 25 Oct 14:05 | none | 25 Oct 15:57, `outputs/v3_urban_s0/esc/fold_1/metrics.json` | the fold-1 precondition heads were scored from 14:42 |
| T18 | `moe/ANALYSIS_PLAN_T18.md` | `1d37676` 25 Oct 14:05 | none | 25 Oct 15:57, `outputs/v3_urban_s0/esc/fold_1/metrics.json` | shares T15's run; the fold-1 precondition heads were scored from 14:42 |
| T17 | `moe/ANALYSIS_PLAN_T17.md` | `ce34331` 25 Oct 14:02 | `8f33aaf` 25 Oct 23:22* | 25 Oct 14:19, `esc50/raw/s0_f1.npz` | amendment 1 covers part (c), before any (c) test clip |
| T16 | `moe/ANALYSIS_PLAN_T16.md` | `55d1560` 25 Oct 14:00 | `8795a3f` 25 Oct 14:06, `89f3f02` 25 Oct 14:47* | 25 Oct 14:42, debugging run on fold 1 in the working folder `t16-data/test/`; the run folder's copies all carry 25 Oct 20:06 | amendment 2 (exploratory PCA routers) came after the fold-1 debugging run (named in the paper) |
| T19 | `moe/ANALYSIS_PLAN_T19.md` | `ae334eb` 26 Oct 14:09 | `0f9958a` 26 Oct 14:09, `0d415db` 26 Oct 19:43 | 26 Oct 19:47, `scores/esc50/L_fp32/s1/fp32__bg__0db.npz` |  |
| T20 | `moe/ANALYSIS_PLAN_T20.md` | `ca935d8` 26 Oct 14:08 | `f415281` 26 Oct 14:18 | 26 Oct 14:22, `runs/urban/f1_s0_lr1e-4.json` |  |
| T21 | `moe/t21/ANALYSIS_PLAN.md` | `731570e` 26 Oct 14:07 | `70a902c` 26 Oct 14:20, `2db1c2d` 26 Oct 14:21, `c2681f7` 26 Oct 20:36*, `d480027` 26 Oct 23:22*, `8e9b4b9` 27 Oct 02:12* | 26 Oct 20:15, `outputs/d1_pool_mn10_s0/esc/fold_1/metrics.json` | amendments 2 (int8 calibration, a secondary) and 3 (FSD50K official split) came after the first FSDKaggle2018 scores and before any FSD50K score |
| T22 | `moe/t22/ANALYSIS_PLAN.md` | `b33df0d` 28 Oct 01:17 | `480c294` 28 Oct 01:22, `d505fce` 28 Oct 01:29, `68ce9c9` 28 Oct 01:29 | 28 Oct 02:14, `runs/esc50_f1_s0_lr1e-4.json` |  |
| T23 | `moe/t23/ANALYSIS_PLAN.md` | `5b913db` 28 Oct 01:21 | `d36d3fe` 28 Oct 06:58* | 28 Oct 02:21, `outputs_p1/wg_esc50_s0/esc/fold_1/metrics.json` | amendment 1 came after the first-stage tables (named in the paper) |
| T24 | `moe/t24/ANALYSIS_PLAN.md` | `9a51cf5` 28 Oct 18:27 | `3724d06` 28 Oct 18:28, `f25b8df` 28 Oct 23:14* | 28 Oct 22:32, `outputs/d3_ast_s0/esc/fold_1/metrics.json` | amendment 2 covers D4 only, after D3 failed its precondition |
| T25 | `moe/t25/ANALYSIS_PLAN.md` | `cc3b048` 28 Oct 22:47 | none | 29 Oct 02:59, `outputs/d1_pool_mn04_s0/esc/fold_1/metrics.json` |  |
| T26 | `moe/t26/ANALYSIS_PLAN.md` | `6ecea17` 29 Oct 09:12 | none | 29 Oct 09:16, `tables/a/verdict.json` | Part A is post hoc and reads T25's stored scores; the first score on new data (Part C, FSDnoisy18k) is 29 Oct 12:48, `outputs/d3_mn20_s0/esc/fold_1/metrics.json` |
| T27 | `moe/t27/ANALYSIS_PLAN.md` | `856efb5` 29 Oct 15:38 | none | 29 Oct 15:40, `tables/d1/by_seed_fold.csv` | post hoc: chosen after T26's results, no pass bar; reads stored scores only; the analysis log starts at 15:38:50 |
| Ideas 1, 2, 3, 4, 5 (stages 1 to 3), 5b, 7; idea 6 stage 1; head-precision headroom; latency; gate by precision; idea 6 exact-length re-test (post hoc) | none in the repository | | | | for the ideas, idea 6 stage 1 and the headroom check, rules in the README of each run folder, written by the run itself; the paper treats idea 6 stage 1 as post hoc. Latency and gate by precision are descriptive, with no pass rule. The exact-length re-test is a follow-up to stage 3c that its plan did not foresee, and its README says nothing in it was pre-registered |

The full-run draft `moe/ANALYSIS_PLAN_T12_FULL_DRAFT.md` was never run. Every plan in the table was committed before
the first scored output of its experiment, with two rows the notes explain: idea 6 stage 2, whose plan covers the noisy
test folds and was written while the clean heads were training, and T12, whose first score time was lost when its run
folder was filled by a copy.

## 6. What this archive does not hold, and what a re-analysis needs

The archive holds code, plans, clip lists and checksums. It holds no audio, embeddings, log-mel caches, trained heads or
score files, so every number needs either the stored run folders or a rerun from audio (section 3 gives the commands and
times). The stored score files of all runs take about 11 GB, and the tables and analysis folders 0.4 GB.

What two checks needed, measured by tracing the files each script opened:

| Check | Files read | Size |
|---|---|---|
| `t21_analyze.py --d d1` (Table 1 FSDKaggle2018 rows, Table 5): `scores/d1/pool_{mn10_w8a8,ast_w4g128}_s{0..4}.npz`, `csv/d1/pool.csv`, `emb/d1/mn_pool/lengths__clean.npy`, `emb/d1/ast_pool/{time_patches,lengths16k}.npy`, and the MN10 weights for the cost model | 14 | 138.8 MB (138.4 MB zipped; float32 logits do not compress) |
| `reviewer_extras/a_ensemble_hull.py` (Table 2): score files of idea 5 stage 2, idea 6 stage 2, 3b and 3c, and stage 3c `analysis/` | 388 | 172.8 MB |

Both reproduced the stored tables to the last digit from a copy of this archive on the CPU (44 s and 8 s). The T21
score files keep every noise condition for both input scalers; the clean and field arrays alone, which the headline
needs, come to 43.9 MB compressed, but `t21_analyze.py` scores all eight conditions and would need a condition option to
run on that subset.

Not in this archive: the stored run folders and Quinn's embeddings (see section 2).
