#!/bin/bash
# T14 pipeline. usage: R=<results dir> bash run_t14.sh   (CPU, nice 10, 2 threads)
set -e
PY=${PY:-python}
cd "$(dirname "$0")"; export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
mkdir -p $R/{inputs,logs,pair,latency,tables}
nice -n 10 $PY t14_check_inputs.py --out $R/inputs > $R/logs/t14_check_inputs.log 2>&1
nice -n 10 $PY t14_pair.py --out $R/pair > $R/logs/t14_pair.log 2>&1
nice -n 10 $PY t14_select.py --out $R/inputs > $R/logs/t14_select.log 2>&1
nice -n 10 $PY t14_primary.py --r $R > $R/logs/t14_primary.log 2>&1
nice -n 10 $PY t14_latency.py --out $R/latency > $R/logs/t14_latency.log 2>&1
nice -n 10 $PY t14_ideas.py --r $R --boot 4000 > $R/logs/t14_ideas.log 2>&1
nice -n 10 $PY t14_singles.py --r $R --boot 2000 > $R/logs/t14_singles.log 2>&1
nice -n 10 $PY t14_report.py --r $R > $R/logs/t14_report.log 2>&1
