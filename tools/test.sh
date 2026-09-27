#!/usr/bin/env bash
# 评测入口（零参可跑）：默认 LIMIT=50 / TRACKER=lqr；ckpt/输出路径由 config + 代码生成（tools/run_config.py）。
# setsid+nohup 后台运行（脱离终端/harness 会话轮换）；[detach]/[exit] 证据写进 logs/stage_eval.log。
# 可用环境变量临时覆盖（同名优先）：CKPT POLICY LIMIT TRACKER WORKERS GPUS DEVICE EXTRA
set -e
GPUS="${GPUS:-0}"; CONFIG="${CONFIG:-config/default.yaml}"
POLICY="${POLICY:-ckpt}"; LIMIT="${LIMIT:-50}"; TRACKER="${TRACKER:-lqr}"
cd "$(dirname "$0")/.."
eval "$(tools/venv-python tools/run_config.py --profile eval --policy "$POLICY" --limit "$LIMIT" --tracker "$TRACKER")"
mkdir -p "$(dirname "$LOG")"
setsid nohup bash -c "
    echo \$\$ > '$DETACH_PID'
    trap 'echo \"[exit] \$(date -Iseconds) signal=TERM\"; exit 143' TERM
    trap 'echo \"[exit] \$(date -Iseconds) signal=INT\"; exit 130' INT
    trap 'echo \"[exit] \$(date -Iseconds) signal=HUP\"; exit 129' HUP
    echo \"[detach] \$(date -Iseconds) pid=\$\$ policy='$POLICY' work_dir='$WORK_DIR'\"
    CUDA_VISIBLE_DEVICES='$GPUS' tools/venv-python tools/test.py --policy '$POLICY' --config '$CONFIG' \
        --spec '${SPEC:-env/specs/scenarios_eval500.json}' --out '$OUT_ROOT' --name '$NAME' \
        --limit '$LIMIT' --workers '${WORKERS:-2}' --tracker '$TRACKER' \
        ${CKPT:+--ckpt '$CKPT'} ${DEVICE:+--device '$DEVICE'} ${EXTRA}
    rc=\$?; echo \"[exit] \$(date -Iseconds) code=\$rc\"; exit \$rc
" >> "$LOG" 2>&1 &
echo "[$(date)] eval $NAME started in background (pid=$!), log: $LOG"
