#!/usr/bin/env python3
"""P2/E-β″ 奖励审计：单 episode rollout 采集器（pre-v6 快照 **或** 当前 HEAD 代码）。

两种代码模式（``--code-mode``）
------------------------------
- ``pre``（默认，P2/E-β′）：审计对象 E-β′（`runs/_refs_rlbase/e_beta_prime/final.pt`，旧
  架构）无法在 HEAD 加载（v6 新头 missing=20），rollout 在 pre-v6 快照（`git archive
  031cc1c` 解包到 /tmp/opencode/v6_pre，见 `tools/reward_audit.py snapshot`）下执行；HEAD
  侧只做离线重放（`tools/reward_audit.py analyze`）。
- ``current``（P4 前置-C，E-β″）：审计对象 = v6 新架构基座（如
  `runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt`），HEAD 代码可直接加载 →
  采集与重放同一份 HEAD 代码（含 P4 前置-B max_step 接线），无需快照。

口径
----
复用 `runs/reward_viz/scripts/rollout_reward_viz.py` 的 rollout + 逐策略步记录管线：
- 闭环执行 = `pipeline.eval_runner._CkptController`（tracker=lqr / eval_reference=plan /
  策略均值动作，与评测口径一致）；
- 奖励调用节奏 = `pipeline.trainer.RewardAdapter.step` 每 0.5 s 策略步一次（dt=0.5），
  ctx 由**真实** `RewardAdapter._build_ctx` 构建（info + post-step obs；info 侧注入
  `lane_lateral_info` + `lane_reward_info` + 实线标志，与 `LocalEnvPool._record` 同口径）；
- 本采集器额外记录**每策略步完整 ctx**（JSON 安全标量），供 HEAD 的新 `RewardAggregator`
  （完整 v5 项集，含 `low_speed`）离线重放；ctx 构建与训练逐位同源，禁用任何近似实现。

用法（由 `tools/reward_audit.py collect` 调起；也可单独运行）::

    PYTHONPATH=/tmp/opencode/v6_pre <venv-python> tools/reward_audit_collect.py \
        --code-mode pre --pre-root /tmp/opencode/v6_pre \
        --ckpt runs/_refs_rlbase/e_beta_prime/final.pt \
        --specs env/specs/scenarios_val.json --spec-id 4 --spec-seed 5000004 \
        --out /tmp/opencode/v6_audit_episodes/id4_seed5000004.json --max-steps 1000

    # E-β″（current-code；在仓库根目录执行）
    <venv-python> tools/reward_audit_collect.py --code-mode current \
        --ckpt runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt \
        --specs env/specs/scenarios_train_5k.json --spec-id 0 --spec-seed 1000 \
        --out /tmp/opencode/ebeta2_episodes/id0_seed1000.json --max-steps 1000

产物：单 episode JSON（meta + 逐策略步 ctx/奖励），并打印一行 `[audit-collect] ...` 摘要。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

__all__ = ["main", "run_collect"]


def _repo_root() -> Path:
    """本采集器所在仓库根目录（`tools/` 的上一级）。"""
    return Path(__file__).resolve().parents[1]


def _head_commit(repo: Path) -> Optional[str]:
    """当前 HEAD commit sha（记录进 meta；git 不可用/非仓库时 None）。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except Exception:  # noqa: BLE001 - 记录性字段，不阻塞采集
        return None
    text = proc.stdout.strip()
    return text if proc.returncode == 0 and text else None


# --------------------------------------------------------------------------- #
# JSON 工具
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) or value is None:
        return value
    return str(value)


def _scalar(value: Any) -> Any:
    """标量 → JSON 安全值；非标量 → None（奖励项对非数值键即按缺键处理）。"""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _sha256_json(payload: Any) -> str:
    text = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def run_collect(args: argparse.Namespace) -> Dict[str, Any]:
    code_mode = str(getattr(args, "code_mode", "pre") or "pre")
    if code_mode not in ("pre", "current"):
        raise SystemExit(f"未知 --code-mode={code_mode!r}（可选 pre|current）")
    root = _repo_root()
    if code_mode == "current":
        # E-β″：采集/重放同一份 HEAD 代码；fail-closed 防误加载快照。
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import pipeline  # noqa: E402

        loaded = Path(pipeline.__file__).resolve()
        if not str(loaded).startswith(str(root)) or "/tmp/opencode/" in str(loaded):
            raise SystemExit(
                f"current-code 模式必须加载仓库 HEAD 代码：pipeline 实际来自 {loaded}（root={root}）"
            )
        pre_root: Optional[Path] = None
        code_root = root
        head_commit: Optional[str] = _head_commit(root)
        code_commit: Optional[str] = head_commit
    else:
        pre_root = Path(args.pre_root).resolve()
        if str(pre_root) not in sys.path:
            sys.path.insert(0, str(pre_root))
        # fail-closed：确认加载的是 pre-v6 快照而不是 HEAD/主仓库
        import pipeline  # noqa: E402

        loaded = Path(pipeline.__file__).resolve()
        if not str(loaded).startswith(str(pre_root)):
            raise SystemExit(f"collector 必须加载 pre-v6 快照：pipeline 实际来自 {loaded}")
        code_root = pre_root
        head_commit = None
        code_commit = str(getattr(args, "pre_commit", "") or "") or None

    from pipeline.eval_runner import (  # noqa: E402
        _CkptController,
        _is_crash,
        _step_dt,
        _unwrap_env,
        load_config,
    )
    from pipeline.trainer import build_reward_adapter, lane_reward_info, resolve_device  # noqa: E402
    from env.metadrive_env import build_env, lane_lateral_info  # noqa: E402
    from env.scenario.spec import load_specs  # noqa: E402

    config = load_config(args.config)
    env_cfg = dict(config.get("env") or {})
    device = resolve_device(args.device, config)
    specs = list(load_specs(args.specs))
    spec = next(
        (s for s in specs if int(s.id) == int(args.spec_id) and int(s.seed) == int(args.spec_seed)),
        None,
    )
    if spec is None:
        raise SystemExit(f"spec id={args.spec_id} seed={args.spec_seed} 不在 {args.specs}")

    task = {
        "spec": spec,
        "policy": "ckpt",
        "ckpt": str(args.ckpt),
        "max_steps": int(args.max_steps),
        "traffic_density": None,
        "policy_params": {},
        "model_config": {
            "hidden_dim": config.get("hidden_dim", 128),
            "moe": dict(config.get("moe") or {}),
            "world_model": dict(config.get("world_model") or {}),
        },
        "obs_config": dict(env_cfg.get("obs") or {}),
        "tracker_config": dict(env_cfg.get("tracking") or {}),
        "device": device,
        "workers": 1,
        "torch_threads": None,
        "omp_num_threads": 1,
        "tracker": str(args.tracker),
        "eval_reference": str(args.eval_reference),
        "moe_off": False,
    }

    # 训练口径奖励适配器（dt=0.5）：包装 factory 抓取每策略步的完整 ctx + StepReward。
    adapter, reward_source = build_reward_adapter(dt=0.5, logger=print)
    captured: List[Dict[str, Any]] = []
    orig_factory = adapter.factory

    def factory_with_capture():
        aggregator = orig_factory()
        orig_step = aggregator.step

        def step_and_capture(ctx, *, step_index=None):
            result = orig_step(ctx, step_index=step_index)
            captured.append({"ctx": dict(ctx), "result": result})
            return result

        aggregator.step = step_and_capture  # type: ignore[method-assign]
        return aggregator

    adapter.factory = factory_with_capture  # type: ignore[assignment]

    def run_reward(info: Mapping[str, Any], obs: Mapping[str, Any], done: bool, step_index: int):
        captured.clear()
        reward, _meta = adapter.step(0, dict(info), obs, done, 0.0, step_index=step_index)
        if not captured:
            raise RuntimeError("未抓到 reward ctx（factory 包装失效）")
        return float(reward), captured[-1]["ctx"], captured[-1]["result"]

    env = _unwrap_env(build_env(spec, traffic_density=None, use_render=False))
    dt = _step_dt(env)
    controller = _CkptController(env, spec, task)
    reset_out = env.reset()
    reset_info: Mapping[str, Any] = {}
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        reset_info = dict(reset_out[1])
    controller.bind(env)
    ego = env.agent

    # builder 包装：复用 controller.action 内的 build（额外 build 会扰动后续行为）
    obs_box: List[Dict[str, Any]] = []
    orig_build = controller.builder.build

    def build_and_capture(env_, spec_):
        obs = orig_build(env_, spec_)
        obs_box.append(
            {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        )
        return obs

    controller.builder.build = build_and_capture  # type: ignore[method-assign]

    def _ld_slot0(obs: Mapping[str, Any]) -> Optional[float]:
        try:
            ld = np.asarray(obs.get("ld"), dtype=np.float64)
            mask = np.asarray(obs.get("ld_mask"), dtype=np.float64).reshape(-1)
            if ld.ndim == 2 and ld.shape[1] >= 5 and mask.size and mask[0] > 0.5:
                return float(ld[0, 4])
        except Exception:  # noqa: BLE001
            pass
        return None

    def _speed_limit_source(
        ctx: Mapping[str, Any], obs: Mapping[str, Any], info: Mapping[str, Any], done: bool
    ) -> str:
        limit = ctx.get("speed_limit_mps")
        if limit is None:
            return "none"
        lane_src = info.get("lane_speed_limit_mps")
        if isinstance(lane_src, (int, float)) and abs(float(lane_src) - float(limit)) < 1e-9:
            return "info.lane_speed_limit_mps"
        ld_limit = _ld_slot0(obs)
        if ld_limit is not None and abs(ld_limit - float(limit)) < 1e-9:
            return "obs.ld_slot0"
        return "terminal_mask_fallback" if done else "unknown"

    def _ctx_record(
        ctx: Mapping[str, Any], obs: Mapping[str, Any], info: Mapping[str, Any], done: bool
    ) -> Dict[str, Any]:
        record = {str(key): _scalar(value) for key, value in ctx.items()}
        record["__speed_limit_source"] = _speed_limit_source(ctx, obs, info, done)
        record["__has_lane_lateral_offset"] = "lane_lateral_offset" in ctx
        record["__has_dist_sides"] = (
            "dist_to_left_side" in ctx and "dist_to_right_side" in ctx
        )
        return record

    records: List[Dict[str, Any]] = []
    policy_step_index = 0
    pending: Optional[Dict[str, Any]] = None
    terminated = truncated = False
    final_info: Dict[str, Any] = dict(reset_info)
    steps = 0
    started = time.time()

    def _settle(obs_post: Mapping[str, Any]) -> None:
        nonlocal policy_step_index, pending
        reward, ctx, result = run_reward(
            pending["info"], obs_post, pending["done"], policy_step_index
        )
        records.append(
            {
                "policy_step": int(policy_step_index),
                "frame": int(pending["frame"]),
                "ctx": _ctx_record(ctx, obs_post, pending["info"], pending["done"]),
                "pre_v6": {
                    "reward": reward,
                    "components": _jsonable(result.components),
                    "dense_sum": float(result.dense_sum),
                    "dense_positive_sum": float(getattr(result, "dense_positive_sum", 0.0)),
                    "dense_negative_sum": float(getattr(result, "dense_negative_sum", 0.0)),
                    "terminating_sum": float(result.terminating_sum),
                    "carl_multiplier": float(result.carl_multiplier),
                    "carl_penalty": float(result.carl_penalty),
                    "terminal_value": float(result.terminal_value),
                    "terminal_key": result.terminal_key,
                    "reason": result.reason,
                    "done": bool(result.done),
                },
            }
        )
        policy_step_index += 1
        pending = None

    for step_index in range(int(args.max_steps)):
        action = controller.action(env)
        if not obs_box:
            raise RuntimeError("controller.action 未调用 ObservationBuilder.build")
        obs_now = obs_box[-1]
        if pending is not None:
            _settle(obs_now)

        _, _, terminated, truncated, info = env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        steps = step_index + 1
        final_info = dict(info)

        # info 注入与 LocalEnvPool._record 同口径
        info_for_reward = dict(info)
        info_for_reward["on_white_continuous_line"] = bool(
            getattr(ego, "on_white_continuous_line", False)
        )
        info_for_reward["on_yellow_continuous_line"] = bool(
            getattr(ego, "on_yellow_continuous_line", False)
        )
        info_for_reward.update(lane_lateral_info(ego))
        info_for_reward.update(lane_reward_info(ego, getattr(env, "engine", None)))

        done = bool(terminated or truncated)
        boundary = ((step_index + 1) % int(round(0.5 / dt)) == 0) or done
        if boundary:
            pending = {"info": info_for_reward, "done": done, "frame": step_index}
        if done:
            break
    if pending is not None:
        # episode 已终局（或达 max_steps）：补一次 build 取终局帧 post-step obs（安全）
        _settle(controller.builder.build(env, spec))

    wall = time.time() - started
    # 分类顺序与 pipeline/eval_runner._run_episode 一致：
    # arrive_dest → collision → out_of_road → max_step（truncated / info.max_step / 达 max_steps）
    reason = (
        "arrive_dest"
        if final_info.get("arrive_dest")
        else "collision"
        if _is_crash(final_info)
        else "out_of_road"
        if final_info.get("out_of_road")
        else "error"
        if final_info.get("error")
        else "max_step"
        if (truncated or bool(final_info.get("max_step")) or steps >= int(args.max_steps))
        else "other"
    )
    ctx_seq = [record["ctx"] for record in records]
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema": "reward_audit_episode/v1",
        "spec_id": int(spec.id),
        "spec_seed": int(spec.seed),
        "spec_blocks": str(getattr(spec, "blocks", "")),
        "spec_geometry": _jsonable(list(getattr(spec, "geometry", None) or [])),
        "spec_primary": str((getattr(spec, "labels", None) or {}).get("geometry", "")),
        "spec_file": str(args.specs),
        "ckpt": str(args.ckpt),
        "ckpt_sha256": _file_sha256(str(args.ckpt)),
        "code_mode": code_mode,
        "code_root": str(code_root),
        "code_commit": code_commit,
        "head_commit": head_commit,
        "pre_root": str(pre_root) if pre_root is not None else None,
        "pre_commit": str(args.pre_commit) if code_mode == "pre" else None,
        "reward_source": reward_source,
        "reward_code_files": {
            rel: _file_sha256(str(code_root / rel))
            for rel in ("reward_model/aggregation.py", "reward_model/terms.py")
        },
        "tracker": str(args.tracker),
        "eval_reference": str(args.eval_reference),
        "max_steps": int(args.max_steps),
        "dt": float(dt),
        "device": device,
        "termination": reason,
        "env_terminated": bool(terminated),
        "env_truncated": bool(truncated),
        "steps": int(steps),
        "policy_steps": int(len(records)),
        "route_completion": _scalar(final_info.get("route_completion")),
        "final_speed_mps": _scalar(final_info.get("velocity")),
        "wall_time_s": wall,
        "ctx_sha256": _sha256_json(ctx_seq),
    }
    env.close()
    return {"meta": meta, "steps": records}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="奖励审计单 episode 采集（pre-v6 快照 / 当前 HEAD 代码）")
    parser.add_argument(
        "--code-mode",
        choices=("pre", "current"),
        default="pre",
        help="pre = pre-v6 快照（E-β′，P2）；current = 当前 HEAD 代码（E-β″，P4 前置-C）",
    )
    parser.add_argument("--pre-root", default=None, help="pre-v6 代码快照根目录（code-mode=pre 必填）")
    parser.add_argument("--pre-commit", default="031cc1c", help="快照 commit（pre 模式写入 meta）")
    parser.add_argument("--specs", required=True)
    parser.add_argument("--spec-id", type=int, required=True)
    parser.add_argument("--spec-seed", type=int, required=True)
    parser.add_argument(
        "--ckpt", default=str(Path.cwd() / "runs/_refs_rlbase/e_beta_prime/final.pt")
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--tracker", default="lqr")
    parser.add_argument("--eval-reference", default="plan")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.code_mode == "pre" and not args.pre_root:
        parser.error("code-mode=pre 需要 --pre-root（快照根目录）")

    doc = run_collect(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(doc), handle, ensure_ascii=False)
    tmp_path.replace(out_path)
    meta = doc["meta"]
    print(
        f"[audit-collect] id={meta['spec_id']} seed={meta['spec_seed']} "
        f"termination={meta['termination']} steps={meta['steps']} "
        f"policy_steps={meta['policy_steps']} wall={meta['wall_time_s']:.1f}s "
        f"ctx_sha256={meta['ctx_sha256'][:20]}… → {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
