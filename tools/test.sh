#!/usr/bin/env bash
# 冻结评测协议入口：`tools/test.py` + 50 条 val slice + 固定 tracker（默认 lqr）。
#
# 用法::
#
#     bash tools/test.sh                                  # ckpt 评测（需要 CKPT）
#     CKPT=runs/train/pipeline_x/stage_b/final.pt bash tools/test.sh
#     POLICY=baseline bash tools/test.sh                  # 规则基线参照
#     LIMIT=10 WORKERS=1 bash tools/test.sh CKPT=...      # 薄切片冒烟
#
# 环境变量（全部可覆盖）：
#   POLICY=ckpt|baseline   CKPT   SPEC=env/specs/scenarios_val_slice50.json
#   LIMIT=50   WORKERS=2   TRACKER=lqr|exact   NAME=<自动>   OUT=runs/eval
#   CONFIG   SEED   EXTRA（附加 CLI 片段）
#
# 保证：`set -euo pipefail`；GL 运行库守卫；git hash/配置/参数写入
# `<OUT>/<NAME>/manifest.txt`；日志同时落 `<OUT>/<NAME>/test.log`。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 位置参数 KEY=VALUE 直接覆盖环境变量：bash tools/test.sh CKPT=... LIMIT=10
for arg in "$@"; do
  case "$arg" in
    *=*) export "$arg" ;;
    *) echo "[test.sh] 忽略位置参数 $arg（请用 KEY=VALUE）" >&2 ;;
  esac
done

POLICY="${POLICY:-ckpt}"
CKPT="${CKPT:-}"
SPEC="${SPEC:-env/specs/scenarios_val_slice50.json}"
LIMIT="${LIMIT:-50}"
WORKERS="${WORKERS:-2}"
TRACKER="${TRACKER:-lqr}"
OUT="${OUT:-runs/eval}"
CONFIG="${CONFIG:-config/default.yaml}"
SEED="${SEED:-0}"
NAME="${NAME:-}"
EXTRA="${EXTRA:-}"

PYTHON="$ROOT/tools/venv-python"
TEST_PY="$ROOT/tools/test.py"

# ---------------------------------------------------------------- 前置检查/GL 守卫
if [[ ! -x "$PYTHON" ]]; then
  echo "[test.sh] 缺少 venv python：$PYTHON" >&2
  exit 2
fi
if [[ ! -f "$SPEC" ]]; then
  echo "[test.sh] 评测 spec 不存在：$SPEC（冻结协议默认 scenarios_val_slice50.json）" >&2
  exit 2
fi
if [[ "$POLICY" == "ckpt" && -z "$CKPT" ]]; then
  echo "[test.sh] POLICY=ckpt 需要 CKPT=<path>（或 POLICY=baseline）" >&2
  exit 2
fi
if [[ "$POLICY" == "ckpt" && ! -f "$CKPT" ]]; then
  echo "[test.sh] checkpoint 不存在：$CKPT" >&2
  exit 2
fi
GLDIR="$ROOT/.venv/gl/usr/lib/x86_64-linux-gnu"
if [[ -e "$GLDIR/libGL.so.1" ]]; then
  export LD_LIBRARY_PATH="$GLDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ -x "$ROOT/tools/setup_gl_libs.sh" ]]; then
  echo "[test.sh] glvnd 缺失 → 运行 tools/setup_gl_libs.sh"
  bash "$ROOT/tools/setup_gl_libs.sh"
  if [[ -e "$GLDIR/libGL.so.1" ]]; then
    export LD_LIBRARY_PATH="$GLDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

TIME_TAG="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$NAME" ]]; then
  if [[ "$POLICY" == "baseline" ]]; then
    NAME="baseline_slice${LIMIT}_${TIME_TAG}"
  else
    NAME="$(basename "$(dirname "$CKPT")")_slice${LIMIT}_${TIME_TAG}"
  fi
fi
RUN_DIR="$OUT/$NAME"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/test.log"

{
  echo "created_at: $(date -Iseconds)"
  echo "protocol: frozen_val_slice (spec=$SPEC, limit=$LIMIT, tracker=$TRACKER, workers=$WORKERS)"
  echo "policy: $POLICY"
  echo "ckpt: ${CKPT:-<none>}"
  echo "git_hash: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty: $(git status --porcelain 2>/dev/null | wc -l) 条目未提交"
  echo "config: $CONFIG"
  echo "seed: $SEED"
  echo "extra: $EXTRA"
} > "$RUN_DIR/manifest.txt"
if [[ -f "$CONFIG" ]]; then cp "$CONFIG" "$RUN_DIR/config.snapshot.yaml"; fi

echo "[test.sh] git $(git rev-parse --short HEAD 2>/dev/null || echo unknown) · policy=$POLICY · tracker=$TRACKER · limit=$LIMIT"
echo "[test.sh] → $RUN_DIR（日志 $LOG）"

# shellcheck disable=SC2086
ckpt_args=()
[[ -n "$CKPT" ]] && ckpt_args=(--ckpt "$CKPT")
"$PYTHON" "$TEST_PY" \
  --policy "$POLICY" --spec "$SPEC" --out "$OUT" --name "$NAME" \
  --config "$CONFIG" --seed "$SEED" \
  --limit "$LIMIT" --workers "$WORKERS" --tracker "$TRACKER" \
  "${ckpt_args[@]}" ${EXTRA} 2>&1 | tee "$LOG"

echo "[test.sh] DONE → $RUN_DIR"
