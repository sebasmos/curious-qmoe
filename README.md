# Uncertainty-routed quantized mixture of experts for audio classification

Code for the submission. Two quantized experts answer each clip: MN10-Q8 (EfficientAT MN10, 8-bit weights and
activations) runs on every clip, and when its own top-1 minus top-2 softmax margin is low, AST-Q4 (Audio Spectrogram
Transformer, 4-bit weights, group size 128) also runs and the two probability vectors are averaged. The router is the
cheap expert's own margin, fixed on clean inner validation; it is not learned.

`REPRODUCE_PAPER.md` maps every result in the paper to the commands that produce it, with dataset downloads,
checksums and the observed runtime on an Apple M2 Pro.

## Layout

| Path | What it holds |
|---|---|
| `QWave/` | library: quantizers (`backbone_quant.py`, `effnet_quant.py`, `bitnnet.py`), EfficientAT MN networks (`efficientat_mn.py`, `efficientat_dymn.py`), AST frame handling (`ast_frames.py`), the cost model (`costs.py`), the int8 deployment scaler (`deploy.py`), PANNs CNN14 and Wavegram-CNN14 (`panns_*.py`) |
| `scripts/benchmark.py` | trains the classifier heads (FP32, 1 to 16 bit, BitNet) under the cross-validation protocol |
| `scripts/redesign/idea6/` | the two-expert design on ESC-50, UrbanSound8K (`stage2/`, `stage3b/`, `stage3c/`, `final/`) and TAU 2020 Mobile (`optionB/`), gate quality by precision (`gatecross/`) |
| `scripts/redesign/moe/t21/` | the pre-registered replication on FSDKaggle2018 and FSD50K |
| `scripts/redesign/moe/`, `scripts/redesign/idea{1,2,3,4,5,5b,7}/` | the negative results (learned routers and gates, precision cascades, more experts, specialization) |
| `scripts/download_datasets.sh`, `scripts/setup_original_layout.sh` | dataset downloads with checksums; the folder layout the original drivers expect |
| `data/meta/` | clip lists with folds and labels for ESC-50 and UrbanSound8K (no audio, no features) |
| `data/folds/` | the clip lists with folds the runs used for FSDKaggle2018, FSD50K, FSDnoisy18k, TUT 2017 and TAU 2020 Mobile (Freesound uploader names replaced by integers) |
| `scripts/audits/` | the code of the separate audit passes and of the fact checks behind a few numbers in the text |
| `environment/` | `pip freeze` of the two environments the runs used |
| `tests/` | unit tests: `PYTHONPATH=. pytest tests/` |

Each stage folder keeps its `ANALYSIS_PLAN.md`, written and committed before any test fold was scored.

## Install

```bash
conda create -n qmoe python=3.11 -y && conda activate qmoe
pip install -r environment/qmoe.lock.txt          # every version the runs used (Python 3.11.14, torch 2.7.0)
# separate environment for the ONNX Runtime int8 path (it reads cached log-mels, so no torchaudio)
conda create -n qmoe-ort python=3.11 -y
conda run -n qmoe-ort pip install -r environment/qmoe-ort.lock.txt   # onnxruntime 1.31.0, torch 2.7.0
PYTHONPATH=. pytest tests/
```

No audio, embeddings, weights or score files ship with this archive. Pretrained weights are fetched from their public
releases; `REPRODUCE_PAPER.md` section 2 lists their URLs, MD5 and SHA-256, and section 6 what a re-analysis from stored
scores would need. `requirements.txt` is the looser list of the original library.

The library still has a loader and `config/quinn.yaml` for Quinn, a dataset this study does not use. Quinn's raw
audio is not public and only its embeddings exist, so nothing in the paper uses it and nothing here can rebuild it.

## Licence

MIT, see `LICENSE`.
