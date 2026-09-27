#!/usr/bin/env bash
# 训练入口（零参可跑）：work_dir/日志/ckpt/数据集等路径由 config + 代码生成（tools/run_config.py），
# 不在此写死。setsid+nohup 后台运行（脱离终端/harness 会话轮换）；存活/退出证据写进 stage 日志。
# 可用环境变量临时覆盖（同名优先）：STAGE RESUME BC_DIR WORK_DIR GPUS DEVICE LIMIT_DATASET EXTRA
set -e
GPUS="${GPUS:-0}"; CONFIG="${CONFIG:-config/default.yaml}"
cd "$(dirname "$0")/.."
eval "$(tools/venv-python tools/run_config.py --profile train --stage "${STAGE:-}")"
mkdir -p "$(dirname "$LOG")"
setsid nohup bash -c "
    echo \$\$ > '$DETACH_PID'
    ( while sleep 30 && kill -0 \$\$ 2>/dev/null; do echo \"[heartbeat] \$(date -Iseconds) last=\$(tail -n1 '$LOG' | cut -c1-160)\"; done ) & hb=\$!
    trap 'echo \"[exit] \$(date -Iseconds) signal=TERM\"; kill \$hb 2>/dev/null; exit 143' TERM
    trap 'echo \"[exit] \$(date -Iseconds) signal=INT\"; kill \$hb 2>/dev/null; exit 130' INT
    trap 'echo \"[exit] \$(date -Iseconds) signal=HUP\"; kill \$hb 2>/dev/null; exit 129' HUP
    echo \"[detach] \$(date -Iseconds) pid=\$\$ stage='$STAGE' work_dir='$WORK_DIR'\"
    CUDA_VISIBLE_DEVICES='$GPUS' tools/venv-python tools/train.py --stage '$STAGE' --config '$CONFIG' \
        --bc-dir '$BC_DIR' --out '$STAGE_DIR' --wm-epochs '$WM_EPOCHS' --bc-epochs '$BC_EPOCHS' --ckpt-every '$CKPT_EVERY' \
        --batch-size '$BATCH_SIZE' --micro-batch-size '$MICRO_BATCH_SIZE' ${TRAJ_AUX_WEIGHT:+--traj-aux-weight '$TRAJ_AUX_WEIGHT'} \
        ${CKPT:+--ckpt '$CKPT'} ${RESUME:+--resume '$RESUME'} ${DEVICE:+--device '$DEVICE'} \
        ${LIMIT_DATASET:+--limit-dataset '$LIMIT_DATASET'} ${EXTRA}
    rc=\$?; kill \$hb 2>/dev/null; echo \"[exit] \$(date -Iseconds) code=\$rc\"; exit \$rc
" >> "$LOG" 2>&1 &
echo "[$(date)] train stage=$STAGE started in background (pid=$!), log: $LOG"
