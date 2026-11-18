#!/usr/bin/env bash
# Supplementary analyses: re-run with per-contract scores (exp_r1.py), then
# aggregate (analyze_r1.py). Cheap models in every cell; TextCNN in every cell as well.
set -uo pipefail
cd "$(dirname "$0")"
export SC_WORK=${SC_WORK:-/home/user/scwork} PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 THREADS=1
LOG=$SC_WORK/logs; mkdir -p "$LOG"
{
  for m in rule lr knn cb; do
    for rg in random dedup cluster external; do echo messi_re $rg $m; done
    for rg in random dedup cluster; do echo messi_ts $rg $m; done
    for rg in official cluster random version external; do echo slither $rg $m; done
  done
  for rg in random dedup cluster external; do echo messi_re $rg cnn; done
  for rg in random dedup cluster; do echo messi_ts $rg cnn; done
  for rg in official cluster random version external; do for s in 0 1 2 3 4 5 6 7 8 9; do echo slither $rg cnn $s; done; done
} | xargs -P ${P:-3} -L 1 bash -c 'python3 exp_r1.py "$@" >> '"$LOG"'/r1.log 2>&1' _
echo R1_DONE >> "$LOG/r1.log"
