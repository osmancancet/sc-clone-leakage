#!/usr/bin/env bash
# Full pipeline. Logs go to $SC_WORK/logs; per-run JSON to results/runs.
# Order: data -> audit -> pool -> embed -> runs -> over-fitting diagnostic ->
# supplementary analyses (per-contract scores, audit checks) -> analyze -> tables/figures.
# The last two analysis steps reuse the splits and seeds of the main runs; they do
# not change run.py or the main-run JSON files.
set -euo pipefail
cd "$(dirname "$0")"
export SC_WORK=${SC_WORK:-/home/user/scwork}
export HF_HUB_DISABLE_XET=1 PYTHONPATH="$PWD"
# one BLAS thread per run; parallelism comes from separate processes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG=$SC_WORK/logs; mkdir -p "$LOG"
stamp() { date -u +%FT%TZ; }

echo "$(stamp) data"   && python3 data.py  > "$LOG/data.log" 2>&1
echo "$(stamp) audit"  && python3 audit.py slither > "$LOG/audit_slither.log" 2>&1
python3 audit.py messi_resnip messi_re messi_ts smartbugs solidifi > "$LOG/audit_small.log" 2>&1
python3 audit_cross.py > "$LOG/audit_cross.log" 2>&1
echo "$(stamp) pool"   && python3 pool.py  > "$LOG/pool.log" 2>&1
echo "$(stamp) embed"  && OMP_NUM_THREADS=4 THREADS=4 python3 embed.py > "$LOG/embed.log" 2>&1

# Messi-Q: 2 datasets x 3 regimes x 5 models x 10 seeds (+ external transfer)
for ds in messi_re messi_ts; do
  for rg in random dedup cluster; do
    for m in rule knn lr cb cnn; do
      THREADS=1 python3 run.py $ds $rg $m >> "$LOG/runs_$ds.log" 2>&1
    done
  done
done
for m in rule knn lr cb cnn; do
  THREADS=1 python3 run.py messi_re external $m >> "$LOG/runs_messi_re.log" 2>&1
done

# Slither pool: 5 regimes x 5 models x 10 seeds, four seeds in parallel
for rg in random cluster official version external; do
  for m in rule knn lr cb cnn; do
    for s in 0 1 2 3 4 5 6 7 8 9; do echo "$rg $m $s"; done
  done
done | xargs -P 4 -L 1 bash -c 'THREADS=1 python3 run.py slither $0 $1 $2' >> "$LOG/runs_slither.log" 2>&1

echo "$(stamp) fit"    && bash fit_jobs.sh              # exp_fit.py -> results/fit/
echo "$(stamp) r1"     && bash r1_jobs.sh               # exp_r1.py  -> results/scores/
python3 audit_r1.py > "$LOG/audit_r1.log" 2>&1           # -> results/audit_r1.json
echo "$(stamp) analyze" && python3 analyze.py > "$LOG/analyze.log" 2>&1
BOOT=2000 python3 analyze_r1.py > "$LOG/analyze_r1.log" 2>&1   # -> results/numbers_r1.json
python3 make_tables.py > "$LOG/tables.log" 2>&1
python3 make_figures.py > "$LOG/figures.log" 2>&1
echo "$(stamp) done"
