#!/usr/bin/env bash
# 训练入口（零参可跑）：路径/ckpt/日志由 config + tools/run_config.py 生成；setsid+nohup 后台化（脱离终端/
# harness 会话轮换），[detach]/[exit] 证据落日志；环境覆盖（同名优先）：STAGE RESUME BC_DIR WORK_DIR GPUS
# DEVICE LIMIT_DATASET EXTRA。
# lane P3-C：PHASE3=1 → A→B→phase3 全链；PHASE3_ONLY=1 → 直接 phase3；PHASE3_ROUNDS=N 覆盖轮数；编排在
# Python（tools/train.py --phase3-chain），本脚本只做后台化/日志/证据行。
set -e; export PYTHONUNBUFFERED=1  # stdout 逐行落盘（否则块缓冲导致日志看起来"卡住"）
GPUS="${GPUS:-0}"; CONFIG="${CONFIG:-config/default.yaml}"
cd "$(dirname "$0")/.."
eval "$(tools/venv-python tools/run_config.py --profile train --stage "${STAGE:-}")"
mkdir -p "$(dirname "$LOG")"
if [ "${PHASE3:-0}" = "1" ] || [ "${PHASE3_ONLY:-0}" = "1" ]; then
    CHAIN_LOG="$(dirname "$LOG")/phase3_chain.log"
    TAG="chain=phase3 only=${PHASE3_ONLY:-0}"
    CMD="CUDA_VISIBLE_DEVICES='$GPUS' tools/venv-python tools/train.py --phase3-chain --config '$CONFIG' ${PHASE3_ROUNDS:+--phase3-rounds '$PHASE3_ROUNDS'} ${DEVICE:+--device '$DEVICE'} ${LIMIT_DATASET:+--limit-dataset '$LIMIT_DATASET'} ${EXTRA}"
else
    CHAIN_LOG="$LOG"
    TAG="stage='$STAGE' work_dir='$WORK_DIR'"
    CMD="CUDA_VISIBLE_DEVICES='$GPUS' tools/venv-python tools/train.py --stage '$STAGE' --config '$CONFIG' --bc-dir '$BC_DIR' --out '$STAGE_DIR' --wm-epochs '$WM_EPOCHS' --bc-epochs '$BC_EPOCHS' --ckpt-every '$CKPT_EVERY' --batch-size '$BATCH_SIZE' --micro-batch-size '$MICRO_BATCH_SIZE' ${TRAJ_AUX_WEIGHT:+--traj-aux-weight '$TRAJ_AUX_WEIGHT'} ${CKPT:+--ckpt '$CKPT'} ${RESUME:+--resume '$RESUME'} ${DEVICE:+--device '$DEVICE'} ${LIMIT_DATASET:+--limit-dataset '$LIMIT_DATASET'} ${EXTRA}"
fi
setsid nohup bash -c "
    echo \$\$ > '$DETACH_PID'
    trap 'echo \"[exit] \$(date -Iseconds) signal=TERM\"; exit 143' TERM
    trap 'echo \"[exit] \$(date -Iseconds) signal=INT\"; exit 130' INT
    trap 'echo \"[exit] \$(date -Iseconds) signal=HUP\"; exit 129' HUP
    echo \"[detach] \$(date -Iseconds) pid=\$\$ $TAG\"
    $CMD
    rc=\$?; echo \"[exit] \$(date -Iseconds) code=\$rc\"; exit \$rc
" >> "$CHAIN_LOG" 2>&1 &
echo "[$(date)] train $TAG started in background (pid=$!), log: $CHAIN_LOG"
