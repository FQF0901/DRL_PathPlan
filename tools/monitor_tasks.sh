#!/usr/bin/env bash
# Decoupled 20-minute task/resource monitor. Safe to run with no active task.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/runs/monitor/tasks_20m.log"
PIDFILE="$ROOT/runs/monitor/tasks_20m.pid"
mkdir -p "$(dirname "$LOG")"
CYCLES="${MONITOR_CYCLES:-72}"   # 72 * 20 min = 24 h
for i in $(seq 1 "$CYCLES"); do
  {
    echo "===== $(date -Is) cycle $i/$CYCLES ====="
    echo "--- gpu ---"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    echo "--- memory/swap ---"
    free -h
    echo "--- active task processes ---"
    ps -eo pid,ppid,etime,pcpu,pmem,rss,cmd --sort=-pcpu \
      | grep -E 'python|metadrive|train_bc|train_ppo|eval_|dataset\.collect|collect_dagger' \
      | grep -v -E 'grep|monitor_tasks' | head -20 || true
    echo "--- latest run artifacts ---"
    find "$ROOT/runs" -name metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 || true
    echo "--- data/run/disk ---"
    du -sh "$ROOT/datasets" "$ROOT/runs" 2>/dev/null || true
    df -h "$ROOT" | tail -1
    echo
  } >> "$LOG" 2>&1
  sleep 1200
done
