#!/usr/bin/env bash
# ============================================================================
# Reproduce every number in the paper, end to end.
#
#   bash scripts/reproduce_paper.sh [ESC_CSV] [QUINN_CSV] [URBAN_CSV]
#
# Defaults point at this machine's dataset layout. Each run writes
# outputs/<tag>/moe/summary.json (or .../<model>/summary.json for singles);
# scripts/collect_paper_numbers.py turns those into the paper's tables.
#
# Runtime: several hours on an Apple-silicon CPU. Runs are independent, so
# they can be launched in parallel or split across machines.
# ============================================================================
set -euo pipefail

# Requires: the `curious-qmoe` conda env (torch, hydra, fvcore, codecarbon) and
# PYTHONPATH set to the repository root, e.g.
#   conda activate curious-qmoe && export PYTHONPATH=$(pwd)

ESC=${1:-/Users/cajas.sebastian/Documents/DATASETS/ESC-50/efficientnet_1536/esc-50.csv}
QUINN=${2:-/Users/cajas.sebastian/Documents/DATASETS/Quinn/efficientnet_ABGQI/ABGQI_embeddings_torch.csv}
URBAN=${3:-/Users/cajas.sebastian/Documents/DATASETS/Urban8k/efficientnet/urbansound8k.csv}
CFG="$(cd "$(dirname "$0")/.." && pwd)/config"
COMMON="experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false"

moe () {  # moe <config> <csv> <experts> <n> <extra...> <tag>
  local cfgname=$1 csv=$2 experts=$3 n=$4 tag=$5; shift 5
  python scripts/benchmark.py --config-path "$CFG" --config-name "$cfgname" \
    $COMMON "experiment.datasets.esc.csv=$csv" \
    "experiment.models_to_run=['moe']" "experiment.router.expert_quantizations=$experts" \
    experiment.router.num_experts=$n experiment.router.top_k=1 \
    "$@" experiment.metadata.tag="$tag"
}

echo "== 1/6 single-model baselines (Table 1, Table 3) =="
for cfgname_csv in "esc50:$ESC" "quinn:$QUINN" "urban8k:$URBAN"; do
  cfgname=${cfgname_csv%%:*}; csv=${cfgname_csv#*:}
  python scripts/benchmark.py --config-path "$CFG" --config-name "$cfgname" \
    $COMMON "experiment.datasets.esc.csv=$csv" \
    "experiment.models_to_run=['1','2','4','8','16','esc','qesc','bitnet']" \
    experiment.metadata.tag="${cfgname}_singles"
done

echo "== 2/6 uniform-routing MoE controls (Table 4 baselines) =="
moe esc50   "$ESC"   "['bitnet','4','8']" 3 esc_uniform_ctrl   experiment.router.use_curiosity=false
moe quinn   "$QUINN" "['bitnet','4','8']" 3 quinn_uniform_ctrl experiment.router.use_curiosity=false
moe urban8k "$URBAN" "['bitnet','4','8']" 3 urban_uniform_ctrl experiment.router.use_curiosity=false

echo "== 3/6 precision prior, alpha sweep + replication (Table 4, alpha sensitivity) =="
for A in 0.3 0.5 1.0; do
  moe esc50 "$ESC" "['bitnet','4','8']" 3 "esc_pprior_a${A}" \
    experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
    experiment.router.curiosity_alpha=$A
done
moe esc50 "$ESC" "['bitnet','4','8']" 3 esc_pprior_a1.0_r2 \
  experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
  experiment.router.curiosity_alpha=1.0
moe quinn   "$QUINN" "['bitnet','4','8']" 3 quinn_pprior_a1.0 \
  experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior experiment.router.curiosity_alpha=1.0
moe urban8k "$URBAN" "['bitnet','4','8']" 3 urban_pprior_a1.0 \
  experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior experiment.router.curiosity_alpha=1.0

echo "== 4/6 SHARED-TRUNK (recommended configuration) =="
for spec in "esc50:$ESC:esc" "quinn:$QUINN:quinn" "urban8k:$URBAN:urban"; do
  cfgname=${spec%%:*}; rest=${spec#*:}; csv=${rest%%:*}; short=${rest#*:}
  moe "$cfgname" "$csv" "['bitnet','4','8']" 3 "${short}_sharedtrunk_a1.0" \
    experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
    experiment.router.curiosity_alpha=1.0 \
    experiment.router.shared_trunk=true experiment.router.trunk_bits=8
done
moe esc50 "$ESC" "['bitnet','4','8']" 3 esc_sharedtrunk_a1.0_r2 \
  experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
  experiment.router.curiosity_alpha=1.0 experiment.router.shared_trunk=true experiment.router.trunk_bits=8

echo "== 5/6 supplementary ablations (Appendix A) =="
for S in kl_divergence entropy_regularization precision_sharp escalation soft_escalation; do
  moe esc50 "$ESC" "['bitnet','4','8']" 3 "esc_${S}_a0.3" \
    experiment.router.use_curiosity=true experiment.router.curiosity_strategy=$S experiment.router.curiosity_alpha=0.3
done
for M in 1 5 20; do
  moe esc50 "$ESC" "['bitnet','4','8']" 3 "esc_pprior_a1.0_mc${M}" \
    experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
    experiment.router.curiosity_alpha=1.0 experiment.router.mc_samples=$M
done
moe esc50 "$ESC" "['bitnet','4','8']" 3 esc_gated_a1.0 \
  experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior \
  experiment.router.curiosity_alpha=1.0 experiment.router.gate_threshold=0.9
for spec in "['bitnet','4','8','16']:4:4experts" "['bitnet','8','16','qesc']:4:8_16_qesc" "['bitnet','qesc']:2:qesc"; do
  experts=${spec%%:*}; rest=${spec#*:}; n=${rest%%:*}; short=${rest#*:}
  moe esc50 "$ESC" "$experts" "$n" "esc_pprior_a1.0_${short}" \
    experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior experiment.router.curiosity_alpha=1.0
done

echo "== 6/6 latency and routing analysis (Section 5.4, Figure 2) =="
python scripts/analysis/latency_benchmark.py --csv "$ESC" --num-passes 100 --mc-samples 10 \
  --device cpu --num-classes 5 --alpha 1.0 --strategy precision_prior \
  --moe-checkpoint outputs/esc_pprior_a1.0/moe/fold_1/best.pth \
  --output-dir scripts/analysis/outputs-paper/latency
python scripts/analysis/routing_analysis.py --csv "$ESC" \
  --checkpoint outputs/esc_pprior_a1.0/moe/fold_1/best.pth \
  --device cpu --curiosity-alpha 1.0 --strategy precision_prior --mc-samples 10 \
  --num-classes 5 --expert-names BitNet Q4 Q8 \
  --output-dir scripts/analysis/outputs-paper/routing

echo
echo "All runs complete. Build the paper tables with:"
echo "  python scripts/collect_paper_numbers.py"
