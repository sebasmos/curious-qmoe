#!/usr/bin/env bash
# ============================================================================
# Every single model under protocol 3 (inner validation split grouped by
# source recording), one scripts/benchmark.py command per (dataset, seed,
# model), for a job queue:
#
#   export ESC_CSV=/path/esc-50.csv QUINN_CSV=/path/ABGQI_embeddings_torch.csv \
#          URBAN_CSV=/path/urbansound8k.csv V3_OUTPUTS=/path/outputs
#   bash scripts/redesign/singles_v3.sh --list > jobs.txt
#   tr '\n' '\0' < jobs.txt | xargs -0 -P 2 -n 1 sh -c
#
# The recipe flags, seeds (ESC-50: 0-4, Quinn and UrbanSound8K: 0-2; override
# with ESC_SEEDS / OTHER_SEEDS) and tags (<ds>_singles_s<seed>) are those of
# the protocol 2 singles in scripts/reproduce_paper.sh --list, plus
# experiment.cross_validation.inner_split_groups=recording. Runs land in
# $V3_OUTPUTS/<tag>/<model kind>/ and logs in $V3_OUTPUTS/logs/. Quinn has no
# recording id, so its inner split stays stratified (the run logs it) while
# its summaries still say protocol 3.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$ROOT/config"
PY="$(command -v python)"
THREADS=2
ESC_SEEDS=${ESC_SEEDS:-"0 1 2 3 4"}
OTHER_SEEDS=${OTHER_SEEDS:-"0 1 2"}
COMMON="experiment.datasets.esc.normalization_type=standard experiment.device=cpu experiment.track_emissions=false experiment.num_threads=$THREADS ${EXTRA_OVERRIDES:-}"
SINGLE_KINDS="1 2 4 8 16 esc qesc bitnet"

sq() { local r="'\\''"; REPLY="'${1//\'/$r}'"; }  # single-quote for sh, builtins only (no forks)

job() {  # job <ds> <seed> <kind>   (prints one command)
  local ds=$1 seed=$2 k=$3 cfgname csv line tag="${1}_singles_s${2}"
  case $ds in
    esc)   cfgname=esc50;   csv=$ESC_CSV ;;
    quinn) cfgname=quinn;   csv=$QUINN_CSV ;;
    urban) cfgname=urban8k; csv=$URBAN_CSV ;;
  esac
  sq "$ROOT"; line="cd $REPLY && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONPATH=$REPLY"
  sq "$PY"; line+=" $REPLY scripts/benchmark.py"
  sq "$CFG"; line+=" --config-path $REPLY --config-name $cfgname $COMMON experiment.seed=$seed"
  sq "experiment.datasets.esc.csv=$csv"; line+=" $REPLY"
  sq "experiment.models_to_run=['$k']"; line+=" $REPLY"
  line+=" experiment.cross_validation.inner_split_groups=recording"
  sq "experiment.output_root=$V3_OUTPUTS"; line+=" $REPLY"
  sq "hydra.run.dir=$V3_OUTPUTS/$tag"; line+=" $REPLY"
  sq "$V3_OUTPUTS/logs/${ds}_singles_${k}_s${seed}.log"; line+=" experiment.metadata.tag=$tag > $REPLY 2>&1"
  echo "$line"
}

list_jobs() {  # longest datasets first, so a parallel pool packs well
  local s k
  : "${ESC_CSV:?set ESC_CSV to the ESC-50 embedding CSV}" "${QUINN_CSV:?set QUINN_CSV to the Quinn embedding CSV}"
  : "${URBAN_CSV:?set URBAN_CSV to the UrbanSound8K embedding CSV}"
  : "${V3_OUTPUTS:?set V3_OUTPUTS to the outputs directory of the protocol 3 runs}"
  mkdir -p "$V3_OUTPUTS/logs"
  for s in $OTHER_SEEDS; do for k in $SINGLE_KINDS; do job urban "$s" "$k"; done; done
  for s in $OTHER_SEEDS; do for k in $SINGLE_KINDS; do job quinn "$s" "$k"; done; done
  for s in $ESC_SEEDS; do for k in $SINGLE_KINDS; do job esc "$s" "$k"; done; done
}

case "${1:-}" in
  --list) list_jobs ;;
  *) echo "usage: $0 --list" >&2; exit 2 ;;
esac
