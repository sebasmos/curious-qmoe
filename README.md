[![arXiv](https://img.shields.io/badge/arXiv-2511.11743-b31b1b.svg)](https://arxiv.org/abs/2511.11743)
[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://github.com/sebasmos/curious-qmoe)

# curious-qmoe

**Quantized audio models are fast but erratic. This makes them predictable.**

Shrink an audio classifier to 4 bits and it keeps almost all of its accuracy.
Run it five times and the accuracy moves around. On an edge device that
variability is the problem: you cannot promise a latency budget you cannot
predict.

`curious-qmoe` routes each input to a quantized expert based on how uncertain
the model is about it. Confident inputs stay on the cheap low-bit expert.
Uncertain ones go to a higher-precision one. The aim is lower cross-fold
variance than uniform routing.

<p align="center">
  <img src="docs/architecture.svg" width="440" alt="Uncertainty-aware routing: a shared 8-bit trunk feeds a Bayesian router whose epistemic uncertainty reweights routing toward higher-precision experts.">
</p>

```python
import torch
from omegaconf import OmegaConf
from QWave.moe import qMoEModelBatched

cfg = OmegaConf.load("config/esc50.yaml")
r = cfg.experiment.router
r.use_curiosity = True                     # uncertainty-aware routing
r.shared_trunk = True                      # one shared trunk, half the parameters
r.expert_quantizations = ["bitnet", 4, 8]
r.num_experts = 3

model = qMoEModelBatched(cfg, in_dim=1536, num_classes=5, num_experts=3, top_k=1)
logits, routing_probs, lb_loss, uncertainty = model(torch.randn(8, 1536))
```

## At a glance

| | |
|---|---|
| Routing | Monte Carlo dropout uncertainty shifts routing toward higher-precision experts |
| Size | Experts share one quantized trunk, so only their small heads are replicated |
| Precision | BitNet ternary, BitLinear 1 to 8 bit, fp16, FP32, int8 post-training quantization |

Measured results are in the paper and are rebuilt by the collector below, so
this page does not repeat them.

**The honest part:** a single 4-bit model is still faster and more accurate than
any mixture we measured. We report that rather than bury it. Use this when
predictability matters more than raw speed.

## Install

```bash
conda create -n curious-qmoe python=3.11 -y && conda activate curious-qmoe
git clone https://github.com/sebasmos/curious-qmoe.git && cd curious-qmoe
pip install -e .
```

## How it works

The router estimates epistemic uncertainty with Monte Carlo dropout, then
reweights its routing distribution toward higher-precision experts in
proportion to it:

$$p_i^{\text{UA}} \propto p_i \cdot \exp(\alpha \, u \, \beta_i)$$

where $u$ is the sample's uncertainty and $\beta_i$ is expert $i$'s normalized
precision. When the model is confident, $u \to 0$ and routing is unchanged.
When it is not, mass shifts toward the experts that can afford to be right.

This produces *polarization*: a trained router sends most samples to one
dominant expert, removing the cost-mode switching that makes mixture latency
jump around.

## Reproducing the paper

The run matrix and the collector below rebuild every table. Setup, the
evaluation protocol, the full run matrix and the claim-to-command map are in
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

```bash
export ESC_CSV=... QUINN_CSV=... URBAN_CSV=...                    # embedding CSVs
bash scripts/reproduce_paper.sh --list > jobs.txt                 # every training run, over seeds
tr '\n' '\0' < jobs.txt | xargs -0 -P 5 -n 1 sh -c                # run them, 5 at a time
bash scripts/reproduce_paper.sh --analysis                        # latency, energy, routing; run alone
python scripts/collect_paper_numbers.py                           # tables and significance tests
```

The experiments reviewers asked for (confidence cascades, mixed-precision and
parameter-matched single models, ESC-50 at 50 classes, an AST backbone) run the
same way from `--list-extra` and `--analysis-extra`. The AST embeddings come
from [data/audio_embeddings](data/audio_embeddings/README.md).

The code behind arXiv v4 is tagged `arxiv-v4`. The protocol has changed since
(predefined folds, model selection on an inner validation split, seeds, macro
F1); CONTRIBUTING describes it.

## Citing

```bibtex
@article{cajas2026uncertainty,
  title   = {Uncertainty-Aware Routing Makes Quantized Mixture-of-Experts Stable},
  author  = {Cajas Ordo{\~n}ez, Sebastian Andres and others},
  journal = {arXiv preprint arXiv:2511.11743},
  year    = {2026}
}
```

MIT licensed.
