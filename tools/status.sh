#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "===== $(date -Is) ====="
echo "git: $(git rev-parse --short HEAD) $(git branch --show-current)"
echo "--- resources ---"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
free -h | head -2
echo "--- active tasks ---"
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'dataset.collect|collect_dagger|pipeline.train_bc|pipeline.train_ppo|pipeline.eval|monitor_tasks' | grep -v grep || true
echo "--- latest logs ---"
for f in runs/logs/collect_v2_2k_shards.log runs/logs/collect_dagger_v2_2k.log runs/logs/train_bc_v2_2k_q_dagger_r2.log runs/monitor/tasks_20m.log; do
  [ -f "$f" ] && { echo "## $f"; tail -5 "$f"; }
done
echo "--- latest metrics ---"
find runs -name metrics.json -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -5 || true
