#!/usr/bin/env bash
# Re-aggregate results, rebuild tables/figures/manuscript, record code digests.
set -euo pipefail
cd "$(dirname "$0")"
LOG=${SC_WORK:-/home/user/scwork}/logs; mkdir -p "$LOG"
{
  date -u +%FT%TZ
  export PYTHONPATH="$PWD"
  python3 analyze.py
  BOOT=${BOOT:-2000} python3 analyze_r1.py | tail -3
  python3 make_tables.py
  python3 make_figures.py
  (cd ../manuscript && python3 build_docx.py)
  echo "--- SHA-256 of code files"
  sha256sum *.py *.sh ../manuscript/build_docx.py | tee ../results/code_sha256.txt
  echo "--- runs: $(ls ../results/runs | wc -l)"
} 2>&1 | tee "$LOG/finalize.log"
cp "$LOG"/*.log ../results/logs/ 2>/dev/null || { mkdir -p ../results/logs; cp "$LOG"/*.log ../results/logs/; }
