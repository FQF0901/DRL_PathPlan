#!/usr/bin/env bash
# 生成场景 spec（P1a 实现）：
#   - 默认：全量 spec -> env/specs/scenarios_{train,val}.json（默认 10000 / 1000 条）
#   - --slice N：额外生成 P1a 切片 -> env/specs/scenarios_{train,val}_slice*.json
#   - 生成后由 env.scenario.cli 惰性调用 validator 做实例化校验（--no-validate 可跳过）
# 用法：bash tools/gene_env.sh [--slice [N]] [--n-train N] [--n-val N] [--out-dir DIR]
#                               [--seed-range A B] [--val-seed-range A B] [--workers W]
#                               [--rng-seed S] [--no-validate]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
用法: bash tools/gene_env.sh [选项]

选项:
  --slice [N]        额外生成 P1a 切片：训练 N 条（默认 200）+ 验证 max(1, N//4) 条
  --n-train N        全量训练 spec 条数（默认 10000）
  --n-val N          全量验证 spec 条数（默认 1000）
  --out-dir DIR      spec 输出目录（默认 env/specs）
  --seed-range A B   训练 seed 区间 [A, B)（默认 [1000, 1000+条数)）
  --val-seed-range A B  验证 seed 区间 [A, B)（默认 [5000000, ...)）
  --workers N        实例化校验并行进程数（默认 8）
  --rng-seed S       生成器随机主种子（默认 0）
  --no-validate      跳过实例化校验
  --out DIR          同 --out-dir（兼容旧参数）
  --seed S           同 --rng-seed（兼容旧参数）
  -h, --help         显示本帮助

环境变量:
  VENV_PYTHON   使用的 Python 解释器（默认 tools/venv-python；缺失时回退 python3）
EOF
}

py="${VENV_PYTHON:-$ROOT/tools/venv-python}"
if [[ ! -x "$py" ]]; then
  echo "[gene_env] 警告: $py 不可执行，回退 python3（spec 生成为纯 Python，不需要 GL）" >&2
  py="python3"
fi

# 参数只做转发与旧参数兼容；--slice 的值可选，其余交给 argparse 校验。
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --out) args+=(--out-dir "${2:?--out 需要参数}"); shift 2 ;;
    --seed) args+=(--rng-seed "${2:?--seed 需要参数}"); shift 2 ;;
    --slice)
      if [[ $# -gt 1 && "${2:-}" =~ ^[0-9]+$ ]]; then
        args+=(--slice "$2"); shift 2
      else
        args+=(--slice); shift 1
      fi
      ;;
    *) args+=("$1"); shift ;;
  esac
done

cd "$ROOT"
exec "$py" -m env.scenario.cli "${args[@]}"
