#!/usr/bin/env bash
# TensorBoard 入口（零参）：run 名由代码生成（train_stageA/train_stageB/eval_stageB/eval_stageA，
# 见 pipeline/run_paths.tb_spec）；端口占用自动 +1；PORT / TB_SPEC 可覆盖。
set -e
cd "$(dirname "$0")/.."
eval "$(tools/venv-python tools/run_config.py --profile tb)"
[[ -n "$TB_SPEC" ]] || { echo "无可用 monitor 事件目录（先跑 tools/train.sh）" >&2; exit 2; }
PORT="${PORT:-6006}"
for _ in $(seq 1 20); do
  tools/venv-python -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" || break
  PORT=$((PORT+1))
done
mkdir -p /tmp/opencode
setsid nohup tools/venv-python -m tensorboard.main --logdir_spec "$TB_SPEC" \
    --port "$PORT" --bind_all --reload_interval 10 > "/tmp/opencode/tb_${PORT}.log" 2>&1 &
echo "TB: http://localhost:${PORT}/  spec: ${TB_SPEC}"
