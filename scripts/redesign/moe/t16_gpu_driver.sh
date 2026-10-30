#!/bin/bash
# Runs the checks then the extractions, one lock-holding process at a time.
P=${PY:-python}
M=$(cd "$(dirname "$0")" && pwd); W=$HOME/Documents/qmoe-work/t16-data; cd $M
export PYTHONPATH=$M/../../.. HF_HUB_OFFLINE=1
for bed in tau urban esc50; do ./lockrun.sh agent-t16-check-$bed nice -n 5 $P t16_extract.py --bed $bed --check > $W/logs/check_$bed.log 2>&1; echo "$(date '+%F %T') check $bed exit $?" >> $W/logs/driver.log; done
for bed in urban esc50 tau; do
  if [ $bed = tau ]; then cells="clean"; else cells=$($P -c "import sys; sys.path.insert(0,'.'); from t16_common import *; print(' '.join(ALL_CELLS['$bed']))"); fi
  for c in $cells; do ./lockrun.sh agent-t16-$bed-$c nice -n 5 $P t16_extract.py --bed $bed --cells $c >> $W/logs/extract_$bed.log 2>&1; echo "$(date '+%F %T') extract $bed $c exit $?" >> $W/logs/driver.log; done
done
echo "$(date '+%F %T') ALL DONE" >> $W/logs/driver.log
