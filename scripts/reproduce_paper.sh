#!/usr/bin/env bash
# ============================================================================
# Reproduce every number in the paper, end to end.
#
# Dataset CSVs come from the environment:
#   export ESC_CSV=/path/esc-50.csv QUINN_CSV=/path/ABGQI_embeddings_torch.csv \
#          URBAN_CSV=/path/urbansound8k.csv
#
# Parallel (recommended): emit the training jobs, one shell command per line,
# run them with xargs, then run the analysis phase ALONE (it times inference,
# so nothing else may share the machine), then build the tables:
#   bash scripts/reproduce_paper.sh --list > jobs.txt
#   tr '\n' '\0' < jobs.txt | xargs -0 -P 5 -n 1 sh -c
#   bash scripts/reproduce_paper.sh --analysis
#   python scripts/collect_paper_numbers.py
#
# Serial: bash scripts/reproduce_paper.sh   (every job, then the analysis)
#
# Every run is repeated over seeds (ESC-50: 0-4, Quinn and UrbanSound8K: 0-2;
# override with ESC_SEEDS / OTHER_SEEDS) and writes
# outputs/<tag>_s<seed>/<model_kind>/summary.json; logs go to outputs/logs/.
# Each job pins OMP/MKL and torch to 2 threads. Folds do not move with the
# seed (predefined folds for ESC-50 and UrbanSound8K, a fixed stratified split
# for Quinn), so every arm and seed is paired on identical test folds.
# EXTRA_OVERRIDES is appended to every job (e.g. experiment.train.max_epochs=2
# for a smoke test).
#
# Reviewer experiments, the same way (also needs AST_ESC_CSV and AST_URBAN_CSV,
# the AST embeddings of ESC-50 and UrbanSound8K, see data/audio_embeddings/):
#   bash scripts/reproduce_paper.sh --list-extra > jobs-extra.txt
#   tr '\n' '\0' < jobs-extra.txt | xargs -0 -P 5 -n 1 sh -c
#   BASE_OUTPUTS=/path/to/main/outputs bash scripts/reproduce_paper.sh --analysis-extra
#   python scripts/collect_paper_numbers.py --outputs outputs /path/to/main/outputs
# BASE_OUTPUTS is where the main matrix wrote its runs, if not ./outputs; the
# cascade and the efficiency references read the single models from there.
#
# Requires the curious-qmoe conda env (torch, hydra, fvcore, codecarbon).
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/config"
PY="$(command -v python)"
THREADS=2
ESC_SEEDS=${ESC_SEEDS:-"0 1 2 3 4"}
OTHER_SEEDS=${OTHER_SEEDS:-"0 1 2"}
LOGS="$ROOT/outputs/logs"
COMMON="experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false experiment.num_threads=$THREADS ${EXTRA_OVERRIDES:-}"
# The uniform control is the SAME Bayesian router as the prior arms with the
# prior switched off (alpha = 0), so router architecture and dropout match.
PRIOR="experiment.router.use_curiosity=true experiment.router.curiosity_strategy=precision_prior"
UNIFORM="$PRIOR experiment.router.curiosity_alpha=0.0"
SHARED="experiment.router.shared_trunk=true experiment.router.trunk_bits=8"
SINGLE_KINDS="1 2 4 8 16 esc qesc bitnet"

sq() { local r="'\\''"; REPLY="'${1//\'/$r}'"; }  # single-quote for sh, builtins only (no forks)

job() {  # job <ds> <seed> <tag> <log name> <overrides...>   (prints one command)
  local ds=$1 seed=$2 tag=$3 log=$4; shift 4
  local cfgname csv o line
  case $ds in
    esc)   cfgname=esc50;   csv=$ESC_CSV ;;
    quinn) cfgname=quinn;   csv=$QUINN_CSV ;;
    urban) cfgname=urban8k; csv=$URBAN_CSV ;;
    esc50c) cfgname=esc50; csv=$ESC_CSV ;;
    ast_esc|ast_esc50c) cfgname=esc50; csv=$AST_ESC_CSV ;;
    ast_urban) cfgname=urban8k; csv=$AST_URBAN_CSV ;;
  esac
  # *50c: ESC-50 on its 50 fine classes (label) instead of the 5 super-categories
  case $ds in *50c) set -- "experiment.datasets.esc.label_col=label" "$@" ;; esac
  sq "$ROOT"; line="cd $REPLY && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONPATH=$REPLY"
  sq "$PY"; line+=" $REPLY scripts/benchmark.py"
  sq "$CFG"; line+=" --config-path $REPLY --config-name $cfgname $COMMON experiment.seed=$seed"
  sq "experiment.datasets.esc.csv=$csv"; line+=" $REPLY"
  for o in "$@"; do sq "$o"; line+=" $REPLY"; done
  sq "$LOGS/${log}_s${seed}.log"; line+=" experiment.metadata.tag=${tag}_s${seed} > $REPLY 2>&1"
  echo "$line"
}

moe() {  # moe <ds> <seed> <tag> <experts> <n> <overrides...>
  local ds=$1 seed=$2 tag=$3 experts=$4 n=$5; shift 5
  job "$ds" "$seed" "$tag" "$tag" "experiment.models_to_run=[moe]" \
    "experiment.router.expert_quantizations=$experts" "experiment.router.num_experts=$n" \
    "experiment.router.top_k=1" "$@"
}

jobs_for() {  # jobs_for <ds> <seed>
  local ds=$1 s=$2 k
  local E3="[bitnet,'4','8']"
  # 1. single-model baselines, one job per model kind
  for k in $SINGLE_KINDS; do
    job "$ds" "$s" "${ds}_singles" "${ds}_singles_${k}" "experiment.models_to_run=['$k']"
  done
  # 2. uniform-routing controls, full experts and shared trunk
  moe "$ds" "$s" "${ds}_uniform_ctrl" "$E3" 3 $UNIFORM
  moe "$ds" "$s" "${ds}_sharedtrunk_uniform_ctrl" "$E3" 3 $UNIFORM $SHARED
  # 3. precision prior (alpha sweep on ESC-50), full experts and shared trunk
  local alphas="1.0"; [ "$ds" = esc ] && alphas="0.3 0.5 1.0"
  for A in $alphas; do
    moe "$ds" "$s" "${ds}_pprior_a${A}" "$E3" 3 $PRIOR "experiment.router.curiosity_alpha=$A"
  done
  moe "$ds" "$s" "${ds}_sharedtrunk_a1.0" "$E3" 3 $PRIOR experiment.router.curiosity_alpha=1.0 $SHARED
  [ "$ds" = esc ] || return 0
  # 4. supplementary ablations, ESC-50 only
  for S in kl_divergence entropy_regularization precision_sharp escalation soft_escalation; do
    moe esc "$s" "esc_${S}_a0.3" "$E3" 3 experiment.router.use_curiosity=true \
      "experiment.router.curiosity_strategy=$S" experiment.router.curiosity_alpha=0.3
  done
  for M in 1 5 20; do
    moe esc "$s" "esc_pprior_a1.0_mc${M}" "$E3" 3 $PRIOR experiment.router.curiosity_alpha=1.0 \
      "experiment.router.mc_samples=$M"
  done
  moe esc "$s" esc_gated_a1.0 "$E3" 3 $PRIOR experiment.router.curiosity_alpha=1.0 \
    experiment.router.gate_threshold=0.9
  moe esc "$s" esc_pprior_a1.0_4experts "[bitnet,'4','8','16']" 4 $PRIOR experiment.router.curiosity_alpha=1.0
  moe esc "$s" esc_pprior_a1.0_8_16_qesc "[bitnet,'8','16',qesc]" 4 $PRIOR experiment.router.curiosity_alpha=1.0
  moe esc "$s" esc_pprior_a1.0_qesc "[bitnet,qesc]" 2 $PRIOR experiment.router.curiosity_alpha=1.0
}

arms_for() {  # arms_for <ds> <seed>: the singles and four MoE arms of the extra datasets
  local ds=$1 s=$2 k
  local E3="[bitnet,'4','8']"
  for k in esc 4 8 bitnet qesc; do
    job "$ds" "$s" "${ds}_singles" "${ds}_singles_${k}" "experiment.models_to_run=['$k']"
  done
  moe "$ds" "$s" "${ds}_uniform_ctrl" "$E3" 3 $UNIFORM
  moe "$ds" "$s" "${ds}_sharedtrunk_uniform_ctrl" "$E3" 3 $UNIFORM $SHARED
  moe "$ds" "$s" "${ds}_pprior_a1.0" "$E3" 3 $PRIOR experiment.router.curiosity_alpha=1.0
  moe "$ds" "$s" "${ds}_sharedtrunk_a1.0" "$E3" 3 $PRIOR experiment.router.curiosity_alpha=1.0 $SHARED
}

controls_for() {  # controls_for <ds> <seed>: mixed-precision and parameter-matched singles
  local ds=$1 s=$2 k
  local E3="[bitnet,'4','8']"
  job "$ds" "$s" "${ds}_singles_mp" "${ds}_singles_mp" "experiment.models_to_run=['mp8_4']"
  # widths are matched to the MoE these router settings describe
  for k in pm8_moe pm32_moe pm8_st pm32_st; do
    job "$ds" "$s" "${ds}_singles_pm" "${ds}_singles_pm_${k}" "experiment.models_to_run=['$k']" \
      "experiment.router.expert_quantizations=$E3" "experiment.router.num_experts=3"
  done
}

list_extra() {  # reviewer experiments, longest first
  local s
  : "${ESC_CSV:?set ESC_CSV to the ESC-50 embedding CSV}" "${QUINN_CSV:?set QUINN_CSV to the Quinn embedding CSV}"
  : "${URBAN_CSV:?set URBAN_CSV to the UrbanSound8K embedding CSV}"
  : "${AST_ESC_CSV:?set AST_ESC_CSV to the ESC-50 AST embedding CSV}"
  : "${AST_URBAN_CSV:?set AST_URBAN_CSV to the UrbanSound8K AST embedding CSV}"
  for s in $OTHER_SEEDS; do controls_for urban "$s"; done
  for s in $OTHER_SEEDS; do arms_for ast_urban "$s"; done
  for s in $OTHER_SEEDS; do controls_for quinn "$s"; done
  for s in $ESC_SEEDS; do
    controls_for esc "$s"; arms_for esc50c "$s"; arms_for ast_esc "$s"; arms_for ast_esc50c "$s"
  done
}

list_jobs() {  # longest datasets first, so a parallel pool packs well
  local s
  : "${ESC_CSV:?set ESC_CSV to the ESC-50 embedding CSV}" "${QUINN_CSV:?set QUINN_CSV to the Quinn embedding CSV}"
  : "${URBAN_CSV:?set URBAN_CSV to the UrbanSound8K embedding CSV}"
  for s in $OTHER_SEEDS; do jobs_for urban "$s"; done
  for s in $OTHER_SEEDS; do jobs_for quinn "$s"; done
  for s in $ESC_SEEDS; do jobs_for esc "$s"; done
}

analysis() {
  # Run alone, after training: it measures latency and energy.
  local O="$ROOT/outputs" A ds
  export PYTHONPATH="$ROOT" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  cd "$ROOT"
  echo "== efficiency: batch-1 latency and energy of trained checkpoints (seed 0, fold 1) =="
  for ds in esc quinn urban; do
    A=(--arm "fp32=$O/${ds}_singles_s0/esc" --arm "int8=$O/${ds}_singles_s0/qesc"
       --arm "uniform=$O/${ds}_uniform_ctrl_s0/moe" --arm "pprior=$O/${ds}_pprior_a1.0_s0/moe"
       --arm "st_uniform=$O/${ds}_sharedtrunk_uniform_ctrl_s0/moe" --arm "st_pprior=$O/${ds}_sharedtrunk_a1.0_s0/moe")
    if [ "$ds" = esc ]; then
      A+=(--arm "gated=$O/esc_gated_a1.0_s0/moe" --arm "mc1=$O/esc_pprior_a1.0_mc1_s0/moe"
          --arm "mc5=$O/esc_pprior_a1.0_mc5_s0/moe" --arm "mc20=$O/esc_pprior_a1.0_mc20_s0/moe"
          --arm "experts_4=$O/esc_pprior_a1.0_4experts_s0/moe"
          --arm "experts_8_16_ptq=$O/esc_pprior_a1.0_8_16_qesc_s0/moe"
          --arm "experts_ptq=$O/esc_pprior_a1.0_qesc_s0/moe")
    fi
    "$PY" scripts/analysis/efficiency_benchmark.py "${A[@]}" --threads $THREADS --out "$O/efficiency/$ds.json"
  done
  echo "== routing on held-out test folds (Figure 2) =="
  for a in 0.3 1.0; do
    "$PY" scripts/analysis/routing_analysis.py --run-dir "$O/esc_pprior_a${a}_s0/moe" \
      --output-dir "scripts/analysis/outputs-paper/routing_a${a}"
  done
}

analysis_extra() {
  # Run alone, after the extra training jobs: the efficiency pass times inference.
  local O="$ROOT/outputs" B="${BASE_OUTPUTS:-$ROOT/outputs}" A ds
  export PYTHONPATH="$ROOT" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  cd "$ROOT"
  echo "== confidence and uncertainty cascades from trained single models, every dataset and backbone =="
  "$PY" scripts/analysis/cascade.py --outputs "$O" "$B" --out "$O/cascade"
  echo "== efficiency: batch-1 latency and energy of the new single models (seed 0, fold 1) =="
  for ds in esc quinn urban; do
    A=(--arm "fp32=$B/${ds}_singles_s0/esc" --arm "int8=$B/${ds}_singles_s0/qesc"
       --arm "mp8_4=$O/${ds}_singles_mp_s0/mp8_4")
    for k in pm8_moe pm32_moe pm8_st pm32_st; do A+=(--arm "$k=$O/${ds}_singles_pm_s0/$k"); done
    "$PY" scripts/analysis/efficiency_benchmark.py "${A[@]}" --threads $THREADS --out "$O/efficiency_extra/$ds.json"
  done
  for ds in ast_esc ast_urban; do
    A=(--arm "fp32=$O/${ds}_singles_s0/esc" --arm "int8=$O/${ds}_singles_s0/qesc"
       --arm "uniform=$O/${ds}_uniform_ctrl_s0/moe" --arm "pprior=$O/${ds}_pprior_a1.0_s0/moe"
       --arm "st_uniform=$O/${ds}_sharedtrunk_uniform_ctrl_s0/moe" --arm "st_pprior=$O/${ds}_sharedtrunk_a1.0_s0/moe")
    "$PY" scripts/analysis/efficiency_benchmark.py "${A[@]}" --threads $THREADS --out "$O/efficiency_extra/$ds.json"
  done
}

mkdir -p "$LOGS"
case "${1:-}" in
  --list) list_jobs ;;
  --analysis) analysis ;;
  --list-extra) list_extra ;;
  --analysis-extra) analysis_extra ;;
  "")
    list_jobs > "$LOGS/jobs.txt"
    while IFS= read -r line; do
      echo "== $(echo "$line" | sed 's/.*metadata.tag=\([^ ]*\).*/\1/')"
      sh -c "$line"
    done < "$LOGS/jobs.txt"
    analysis
    echo; echo "All runs complete. Build the paper tables with:"
    echo "  python scripts/collect_paper_numbers.py" ;;
  *) echo "usage: $0 [--list | --analysis | --list-extra | --analysis-extra]" >&2; exit 2 ;;
esac
