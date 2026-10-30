#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-$(cd "$HERE/../../../../.." && pwd)}  # repository root
RESULTS=${RESULTS:-$HOME/Documents/qmoe-results}  # folder that holds one folder per run
E=${E:-$HOME/Documents/DATASETS}  # datasets
QW=${QW:-$HOME/Documents/qmoe-weights}  # EfficientAT and PANNs weights
GUARD=${GUARD:-/dev/null}  # battery guard of the original run, not shipped
# Resume the paused CNN UrbanSound8K noisy processes (10604, 10606) once the AST extraction chain has ended.
R=$RESULTS/2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db
until grep -q "all done" $R/logs/extract_ast.done; do sleep 30; done
kill -CONT 10604 10606 && echo "$(date '+%F %T') AST chain ended; resumed CNN UrbanSound8K noisy processes 10604 10606" >> $R/logs/governor.log
