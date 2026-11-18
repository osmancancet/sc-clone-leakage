#!/usr/bin/env bash
# Over-fitting diagnostic jobs (train / val / test MCC), run after the main runs.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 THREADS=1
LOG=${SC_WORK:-/home/user/scwork}/logs
{
  for ds in messi_re messi_ts; do for rg in random cluster; do for m in lr cnn; do
    for s in 0 1 2 3 4 5 6 7 8 9; do echo $ds $rg $m $s; done; done; done; done
  for rg in official cluster; do for m in lr cnn; do
    for s in 0 1 2 3 4 5 6 7 8 9; do echo slither $rg $m $s; done; done; done
} | xargs -P 4 -L 1 bash -c 'python3 exp_fit.py "$0" "$1" "$2" "$3" >> '"$LOG"'/fit.log 2>&1'
echo FIT_DONE >> "$LOG/fit.log"
