#!/usr/bin/env bash
# Resumable driver for the embedding and run stages (run.py skips finished
# cells, embed.py resumes from saved chunks). Used after a container restart.
set -uo pipefail
cd "$(dirname "$0")"
export SC_WORK=${SC_WORK:-/home/user/scwork} HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 THREADS=1
export LOG=$SC_WORK/logs; mkdir -p "$LOG"

OMP_NUM_THREADS=3 THREADS=3 python3 embed.py >> "$LOG/embed.log" 2>&1 &
EMB=$!

job() {  # dataset regime model seed
  if [ "$1" = slither ] && [ "$3" = cb ]; then
    until [ -f "$SC_WORK/processed/emb_slither.npz" ]; do sleep 30; done
  fi
  python3 run.py "$1" "$2" "$3" "$4" >> "$LOG/runs_$1.log" 2>&1
}
export -f job

{
  for rg in random dedup cluster; do for m in rule knn lr cb cnn; do for s in 0 1 2 3 4 5 6 7 8 9; do
    echo messi_ts $rg $m $s; done; done; done
  for m in rule knn lr cb cnn; do for s in 0 1 2 3 4 5 6 7 8 9; do echo messi_re external $m $s; done; done
  for m in rule lr knn cnn cb; do for rg in official cluster random version external; do
    for s in 0 1 2 3 4 5 6 7 8 9; do echo slither $rg $m $s; done; done; done
} > "$SC_WORK/jobs.txt"
xargs -P 2 -L 1 bash -c 'job "$0" "$1" "$2" "$3"' < "$SC_WORK/jobs.txt" &
XA=$!
wait $EMB
# embedding done: GNU xargs raises its parallelism by one per SIGUSR1
kill -USR1 $XA; kill -USR1 $XA
wait $XA
echo ALL_RUNS_DONE >> "$LOG/runs_slither.log"
