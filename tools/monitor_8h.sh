#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/runs/monitor/monitor_8h.log"
mkdir -p "$(dirname "$LOG")"
for i in $(seq 1 24); do
  {
    echo "===== $(date -Is) cycle $i/24 ====="
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    free -h
    echo '--- top cpu ---'
    ps -eo pid,pcpu,pmem,etime,cmd --sort=-pcpu | head -12
    echo '--- data/run sizes ---'
    du -sh "$ROOT/datasets" "$ROOT/runs" 2>/dev/null || true
    echo '--- latest metrics ---'
    find "$ROOT/runs" -name metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5
    echo
  } >> "$LOG" 2>&1
  sleep 1200
done
