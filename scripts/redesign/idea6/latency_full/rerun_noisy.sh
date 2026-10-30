#!/bin/bash
# Reruns (tag _r2) of the blocks whose first run met a load burst from other agents' jobs: the 2-thread AST blocks
# (load up to 19 to 20 during the block), the 2-thread ONNX Runtime set and the 4-thread torch CNN blocks. The first
# runs are kept; cascade_latency.py takes, per step and thread count, the block with the lowest recorded load.
D=${R:-/Users/cajas.sebastian/Documents/qmoe-results/2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6}   # this run folder
PY=${PY:-python}
PYO=${PYO:-python}
cd $D
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "$(date '+%F %T') t2 torch ast rerun start, uptime: $(uptime)" >> logs/timing.done
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  $PY scripts/time_torch.py --threads 2 --only "ast_*" --tag _r2 > logs/time_torch_t2_r2.log 2>&1
echo "$(date '+%F %T') t2 torch ast rerun exit $?, uptime: $(uptime)" >> logs/timing.done
OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 $PYO scripts/time_ort.py --threads 2 --tag _r2 > logs/time_ort_t2_r2.log 2>&1
echo "$(date '+%F %T') t2 ort rerun exit $?, uptime: $(uptime)" >> logs/timing.done
OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  $PY scripts/time_torch.py --threads 4 --only "effnet_*" "mn*_torch_fp32" --tag _r2 > logs/time_torch_t4_r2.log 2>&1
echo "$(date '+%F %T') t4 torch cnn rerun exit $?, uptime: $(uptime)" >> logs/timing.done
echo "$(date '+%F %T') reruns done" >> logs/timing.done
