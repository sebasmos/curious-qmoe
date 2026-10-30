#!/bin/bash
# Recreates the folder layout that the drivers of the original runs expect, so they run unchanged on another machine:
#   ~/Documents/qmoe-work/<tNN>   links to this repository (the drivers cd into their own worktree)
#   ~/Documents/qmoe-work/*_gpu_run.sh   stand-ins for the shared Apple-GPU lock of the original machine: no lock,
#                                         the command runs directly (fair_gpu_run.sh drops its first argument, a tag)
#   ~/Documents/qmoe-work/t16-data, t15-cache   working folders of the T16 and T15 drivers
#   ~/Documents/qmoe-results, ~/Documents/DATASETS, ~/Documents/qmoe-weights   created when missing
# Existing files and links are never replaced.  Usage: bash scripts/setup_original_layout.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
Q=$HOME/Documents/qmoe-work
mkdir -p "$Q/t23-scratch" "$Q/t16-data" "$Q/t15-cache" "$HOME/Documents/qmoe-results" "$HOME/Documents/DATASETS" "$HOME/Documents/qmoe-weights"
for t in t17 t19 t20 t21 t22 t23 t24 t25 t26; do [ -e "$Q/$t" ] || ln -s "$ROOT" "$Q/$t"; done
if [ ! -e "$Q/fair_gpu_run.sh" ]; then
  printf '#!/bin/bash\n# stand-in for the shared GPU lock: drop the tag, run the command\nshift; exec "$@"\n' > "$Q/fair_gpu_run.sh"
fi
for t in t19 t20 t21; do
  [ -e "$Q/${t}_gpu_run.sh" ] || printf '#!/bin/bash\n# stand-in for the GPU lock wrapper: run the command\nexec "$@"\n' > "$Q/${t}_gpu_run.sh"
done
chmod +x "$Q"/*_gpu_run.sh
echo "layout ready under $Q; for T23 and T15 write their results folder paths into $Q/t23-R.txt and $Q/t15-R.txt"
