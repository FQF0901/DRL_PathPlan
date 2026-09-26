#!/usr/bin/env bash
# 分阶段训练入口（P2/N5）：默认串行跑 **Stage A → Stage B**，全部日志落盘。
#
# 用法::
#
#     bash tools/train.sh                      # A→B（默认 runs/bc_expert_full）
#     STAGE=A bash tools/train.sh              # 只跑 Stage A
#     STAGE=B bash tools/train.sh              # 只跑 Stage B（用 OUT 下已有 A 产物）
#     LIMIT_DATASET=2000 WM_EPOCHS=1 BC_EPOCHS=1 bash tools/train.sh   # 冒烟
#     bash tools/train.sh STAGE=A WM_EPOCHS=1                           # KEY=VALUE 位置参数
#     EXTRA_B="--traj-aux-weight 0.3" bash tools/train.sh
#
# 环境变量（全部可覆盖）：
#   STAGE=both|A|B   CONFIG   MODEL_CONFIG   BC_DIR=runs/bc_expert_full
#   OUT=runs/train/pipeline_<时间戳>   WM_EPOCHS   BC_EPOCHS   BATCH_SIZE
#   LIMIT_DATASET    DEVICE    SEED    EXTRA_A / EXTRA_B（附加 CLI 片段，会做简单分词）
#
# 保证：`set -euo pipefail`；GL 运行库守卫（缺 libGL 时先跑 tools/setup_gl_libs.sh）；
# 记录 git hash + 配置副本 + 环境到 `<OUT>/manifest.txt`；每阶段 stdout/stderr 同时
# 写 `<OUT>/logs/stage_{a,b}.log`。**不启动 Stage C**（EXPERIMENTAL，见 docs/db44fefe-system-review.md）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 位置参数 KEY=VALUE 直接覆盖环境变量：bash tools/train.sh STAGE=A WM_EPOCHS=1
for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *) echo "[train.sh] 忽略位置参数 $arg（请用 KEY=VALUE）" >&2 ;;
  esac
done

STAGE="${STAGE:-both}"
CONFIG="${CONFIG:-config/default.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-config/model.yaml}"
BC_DIR="${BC_DIR:-runs/bc_expert_full}"
OUT="${OUT:-runs/train/pipeline_$(date +%Y%m%d_%H%M%S)}"
WM_EPOCHS="${WM_EPOCHS:-10}"
BC_EPOCHS="${BC_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-}"
LIMIT_DATASET="${LIMIT_DATASET:-}"
DEVICE="${DEVICE:-}"
SEED="${SEED:-0}"
EXTRA_A="${EXTRA_A:-}"
EXTRA_B="${EXTRA_B:-}"

PYTHON="$ROOT/tools/venv-python"
TRAIN_PY="$ROOT/tools/train.py"

# ---------------------------------------------------------------- 前置检查/GL 守卫
if [[ ! -x "$PYTHON" ]]; then
  echo "[train.sh] 缺少 venv python：$PYTHON（先建 .venv）" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "[train.sh] 配置不存在：$CONFIG" >&2
  exit 2
fi
if [[ "$STAGE" != "B" && ! -d "$BC_DIR" ]]; then
  echo "[train.sh] BC 数据集目录不存在：$BC_DIR（用 BC_DIR=... 覆盖）" >&2
  exit 2
fi
GLDIR="$ROOT/.venv/gl/usr/lib/x86_64-linux-gnu"
if [[ -e "$GLDIR/libGL.so.1" ]]; then
  export LD_LIBRARY_PATH="$GLDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ -x "$ROOT/tools/setup_gl_libs.sh" ]]; then
  echo "[train.sh] glvnd 缺失 → 运行 tools/setup_gl_libs.sh（一次性解包到 .venv/gl）"
  bash "$ROOT/tools/setup_gl_libs.sh"
  if [[ -e "$GLDIR/libGL.so.1" ]]; then
    export LD_LIBRARY_PATH="$GLDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  else
    echo "[train.sh] 警告：GL 运行库仍缺失；MetaDrive 可能需要显示环境" >&2
  fi
fi
# torch 线程纪律（避免与其它 env-heavy 任务抢核；stages 内部也会按 config 设）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "$OUT/logs"

# ---------------------------------------------------------------- manifest（配置 + git hash）
{
  echo "created_at: $(date -Iseconds)"
  echo "stage: $STAGE"
  echo "git_hash: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty: $(git status --porcelain 2>/dev/null | wc -l) 条目未提交"
  echo "config: $CONFIG"
  echo "model_config: $MODEL_CONFIG"
  echo "bc_dir: $BC_DIR"
  echo "out: $OUT"
  echo "wm_epochs: $WM_EPOCHS"
  echo "bc_epochs: $BC_EPOCHS"
  echo "batch_size: ${BATCH_SIZE:-<default>}"
  echo "limit_dataset: ${LIMIT_DATASET:-<none>}"
  echo "device: ${DEVICE:-<auto>}"
  echo "seed: $SEED"
  echo "extra_A: $EXTRA_A"
  echo "extra_B: $EXTRA_B"
  echo "python: $("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown)"
  echo "torch: $("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
  echo "cuda_available: $("$PYTHON" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo unknown)"
} > "$OUT/manifest.txt"
if [[ -f "$CONFIG" ]]; then cp "$CONFIG" "$OUT/config.snapshot.yaml"; fi
if [[ -f "$MODEL_CONFIG" ]]; then cp "$MODEL_CONFIG" "$OUT/model.snapshot.yaml"; fi
echo "[train.sh] manifest → $OUT/manifest.txt"
echo "[train.sh] git $(git rev-parse --short HEAD 2>/dev/null || echo unknown) · stage=$STAGE · out=$OUT"

# ---------------------------------------------------------------- 阶段执行
common_args=(
  --config "$CONFIG" --model-config "$MODEL_CONFIG" --bc-dir "$BC_DIR"
  --seed "$SEED" --monitor
)
[[ -n "$BATCH_SIZE" ]] && common_args+=(--batch-size "$BATCH_SIZE")
[[ -n "$LIMIT_DATASET" ]] && common_args+=(--limit-dataset "$LIMIT_DATASET")
[[ -n "$DEVICE" ]] && common_args+=(--device "$DEVICE")

run_stage_a() {
  local stage_out="$OUT/stage_a"
  local log="$OUT/logs/stage_a.log"
  echo "[train.sh] ===== Stage A（WM teacher forcing；$WM_EPOCHS epochs）→ $stage_out ====="
  # shellcheck disable=SC2086
  "$PYTHON" "$TRAIN_PY" --stage A --out "$stage_out" --wm-epochs "$WM_EPOCHS" \
    "${common_args[@]}" ${EXTRA_A} 2>&1 | tee "$log"
}

run_stage_b() {
  local stage_a_out="${STAGE_A_OUT:-$OUT/stage_a}"
  local stage_out="$OUT/stage_b"
  local log="$OUT/logs/stage_b.log"
  local ckpt="$stage_a_out/final.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[train.sh] 警告：Stage A 产物缺失（$ckpt）→ Stage B 将从随机初始化开始" >&2
    ckpt=""
  fi
  echo "[train.sh] ===== Stage B（planner BC；$BC_EPOCHS epochs）→ $stage_out ====="
  local ckpt_args=()
  [[ -n "$ckpt" ]] && ckpt_args=(--ckpt "$ckpt")
  # shellcheck disable=SC2086
  "$PYTHON" "$TRAIN_PY" --stage B --out "$stage_out" --bc-epochs "$BC_EPOCHS" \
    "${ckpt_args[@]}" "${common_args[@]}" ${EXTRA_B} 2>&1 | tee "$log"
}

case "$STAGE" in
  A) run_stage_a ;;
  B) run_stage_b ;;
  both) run_stage_a; run_stage_b ;;
  *) echo "[train.sh] STAGE 必须是 both|A|B（收到 $STAGE）" >&2; exit 2 ;;
esac

echo "[train.sh] DONE → $OUT（Stage C 为 EXPERIMENTAL，不在本脚本内运行）"
