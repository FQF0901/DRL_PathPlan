#!/usr/bin/env python3
"""L2 冒烟：观测通道 shape/mask + 单步耗时分解（obs / physics / traffic / reward）。

用途：
  1. 验证 build_env（显式 blocks、建图后限速、behaviors 钩子）与 ObservationBuilder 全链路可用；
  2. 打印每通道 shape / mask 统计、6 帧历史的 valid 掩码、限速是否已写入（<1000）；
  3. 输出单步耗时分解，供 P1a 的"真实配比吞吐"复测（P0 报告 §5）。

耗时口径（仅测量脚本里做实例级 hook，不改库代码）：
  - obs     = ObservationBuilder.build 的墙钟时间（含 OD/LD/nav/ego/signal + 历史对齐）；
  - traffic = env.engine.traffic_manager.before_step/after_step（IDM 决策 + 触发式生成）；
  - reward  = env.reward_function（MetaDrive 奖励计算）；
  - physics = env.step 总时间 - traffic - reward（物理步进、车辆状态更新、导航、info 组装等）。

用法：
    tools/venv-python tools/measure/obs_smoke.py --seed 1000 --density 0.2 --steps 60
    # 默认用 L1a generator 的 spec（含脚本事件）；--blocks SCXRO 用显式序列兜底 spec；
    # --no-events 清空 cut-in/cut-out 以便长时间跑吞吐；--lru 2 打开地图 LRU。
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fallback_spec(seed: int, blocks: str, density: float):
    """L1a 未就绪时的最小兜底 spec（字段形态与 ScenarioSpec 一致，L2 的宽容读取可跑通）。"""
    chars = list(dict.fromkeys(blocks))
    by_char = {char: (13.9 if char in "SrR" else 8.3) for char in chars}
    return types.SimpleNamespace(
        id=seed,
        seed=seed,
        split="val",
        blocks=blocks,
        geometry=[],
        traffic={"density": density, "patterns": [], "events": [], "seed": seed, "random_traffic": False},
        limits={"speed_limit_mps": min(by_char.values()), "by_char": by_char},
        nav={},
        ego={"spawn_lane_index": None, "spawn_longitude": 5.0, "spawn_lateral": 0.0, "spawn_velocity": 6.0},
        difficulty="easy",
        labels={},
    )


def make_spec(seed: int, blocks: str | None, density: float):
    """优先用 L1a generator 的真实 spec（含校验）；显式 --blocks 或不可用时用兜底 spec。"""
    if blocks is None:
        try:
            from env.scenario.generator import build_specs

            specs, _ = build_specs(1, 0, (int(seed), int(seed) + 1), (int(seed), int(seed) + 1), rng_seed=0)
            if specs:
                return specs[0]
        except Exception as exc:  # noqa: BLE001 - L1a 未就绪/字段变化：退化到兜底 spec
            print(f"[obs_smoke] generator 不可用（{type(exc).__name__}: {exc}），改用兜底 spec")
    return _fallback_spec(seed, blocks or "SCXRO", density)


def instrument(env) -> dict[str, float]:
    """实例级计时 hook（仅本脚本用）：traffic 与 reward 的耗时累加。

    注意：必须在 ``env.reset()``（lazy_init）之后调用，engine/managers 才存在。
    """
    stats = {"traffic": 0.0, "reward": 0.0}
    traffic_manager = env.engine.traffic_manager
    original_before, original_after = traffic_manager.before_step, traffic_manager.after_step

    def timed_before(*args, **kwargs):
        start = time.perf_counter()
        out = original_before(*args, **kwargs)
        stats["traffic"] += time.perf_counter() - start
        return out

    def timed_after(*args, **kwargs):
        start = time.perf_counter()
        out = original_after(*args, **kwargs)
        stats["traffic"] += time.perf_counter() - start
        return out

    traffic_manager.before_step = timed_before
    traffic_manager.after_step = timed_after

    original_reward = env.reward_function

    def timed_reward(*args, **kwargs):
        start = time.perf_counter()
        out = original_reward(*args, **kwargs)
        stats["reward"] += time.perf_counter() - start
        return out

    env.reward_function = timed_reward
    return stats


def print_channel_table(built: dict, title: str) -> None:
    print(f"--- {title} ---")
    print(f"{'channel':<12} {'shape':<14} {'mask_mean':>9} {'valid':>6}  note")
    for name in sorted(k for k in built if not k.endswith("_mask") and not k.endswith("_hist") and k != "hist_valid"):
        feats = built[name]
        mask = built.get(f"{name}_mask")
        mask_mean = float(mask.mean()) if mask is not None else float("nan")
        valid = int((mask > 0.5).sum()) if mask is not None else 0
        note = ""
        if mask is not None and valid == 0:
            note = "no valid slot"
        print(f"{name:<12} {str(tuple(feats.shape)):<14} {mask_mean:>9.3f} {valid:>6}  {note}")
    for key in ("od_hist", "ld_hist"):
        if key in built:
            print(f"{key:<12} {str(tuple(built[key].shape)):<14} {'':>9} {'':>6}  aligned to current ego frame")
    for key in ("od_hist_mask", "ld_hist_mask"):
        if key in built:
            print(f"{key:<12} {str(tuple(built[key].shape)):<14} mean={float(built[key].mean()):.3f}")
    if "hist_valid" in built:
        print(f"{'hist_valid':<12} {str(tuple(built['hist_valid'].shape)):<14} {built['hist_valid']}")


def speed_limit_report(env) -> None:
    try:
        lanes = list(env.current_map.road_network.get_all_lanes())
    except Exception as exc:  # noqa: BLE001
        print(f"speed_limit: 读取失败 {exc}")
        return
    limits = [float(lane.speed_limit) for lane in lanes]
    if not limits:
        print("speed_limit: 地图上没有 lane（异常）")
        return
    unset = sum(1 for v in limits if v >= 1000.0)
    print(
        f"speed_limit(m/s): lanes={len(limits)} set={len(limits) - unset} unset(>=1000)={unset} "
        f"min={min(limits):.2f} max={max(limits):.2f}"
    )
    if unset:
        print("  !! 存在未设置限速的 lane（spec.limits 未覆盖，validator 应报告）")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 obs smoke: channel shapes/masks + step-time breakdown")
    parser.add_argument("--blocks", default=None, help="显式 block 序列；默认用 L1a generator 生成的 spec")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--throttle", type=float, default=0.3)
    parser.add_argument("--lru", type=int, default=0, help="地图 LRU 容量（0=关，≤32）")
    parser.add_argument("--no-events", action="store_true", help="去掉脚本化 cut-in/cut-out（仅测量用，避免短 rollout 被事件车碰撞打断）")
    args = parser.parse_args()

    from env.metadrive_env import build_env
    from env.obs.builder import ObservationBuilder

    spec = make_spec(args.seed, args.blocks, args.density)
    if args.no_events:
        spec_traffic = getattr(spec, "traffic", None)
        if isinstance(spec_traffic, dict):  # 就地清空事件（spec 是本脚本刚构造的，不会被别处复用）
            spec_traffic["patterns"] = []
            spec_traffic["events"] = []
    spec_traffic = getattr(spec, "traffic", None)
    spec_density = spec_traffic.get("density") if isinstance(spec_traffic, dict) else spec_traffic
    spec_patterns = spec_traffic.get("patterns") if isinstance(spec_traffic, dict) else None
    print(
        f"spec: id={getattr(spec, 'id', '?')} seed={getattr(spec, 'seed', '?')} "
        f"blocks={getattr(spec, 'blocks', '?')} density={spec_density} patterns={spec_patterns}"
    )
    print(f"spec.limits={getattr(spec, 'limits', None)}")
    print(f"spec.ego={getattr(spec, 'ego', None)}")
    env = build_env(spec, traffic_density=args.density, lru_size=args.lru)
    builder = ObservationBuilder({"topk_objects": 16, "topk_lanes": 16, "history_frames": 6})
    print(f"feature_spec: {builder.feature_spec}")
    print(f"memory: frames={builder.memory.frames} interval={builder.memory.interval} channels={builder.memory.channels}")

    stats = {"traffic": 0.0, "reward": 0.0}
    totals = {"obs": 0.0, "step": 0.0, "traffic": 0.0, "reward": 0.0}
    nav_commands: dict[str, int] = {}
    n_steps = 0

    try:
        env.reset()  # lazy_init：engine/managers 在这里才被创建
        stats = instrument(env)
        print(f"behavior_install_error: {env.behavior_install_error}")
        print(f"postprocess_stats: {env.postprocess_stats}")
        speed_limit_report(env)

        built = builder.build(env, spec)
        print_channel_table(built, "after reset (episode_step=0)")

        for step in range(args.steps):
            t0 = time.perf_counter()
            _, _, terminated, truncated, info = env.step([args.steer, args.throttle])
            totals["step"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            built = builder.build(env, spec)
            totals["obs"] += time.perf_counter() - t0

            nav_commands[str(info.get("navigation_command"))] = nav_commands.get(str(info.get("navigation_command")), 0) + 1
            n_steps += 1
            if step in (0, 4, 9, 24):
                print(f"step={step + 1:>3} hist_valid={built['hist_valid']}")
            if terminated or truncated:
                print(f"episode ended at step {step + 1} (terminated={terminated} truncated={truncated})")
                break

        print_channel_table(built, f"last step (episode_step={env.episode_step})")
        print(f"nav_commands: {nav_commands}")
        print(f"objects: {len(env.engine.get_objects())}")
    finally:
        env.close()

    if n_steps == 0:
        print("没有有效 step，无法给出耗时分解")
        return
    totals["traffic"] = stats["traffic"]
    totals["reward"] = stats["reward"]
    totals["physics"] = max(0.0, totals["step"] - totals["traffic"] - totals["reward"])
    per_step = {k: v / n_steps for k, v in totals.items()}
    print("--- per-step time breakdown (ms, mean over %d steps) ---" % n_steps)
    for key in ("obs", "physics", "traffic", "reward", "step"):
        print(f"  {key:<8} {per_step[key] * 1e3:8.3f} ms")
    print(f"  fps(env.step only) = {1.0 / per_step['step']:.1f}")


if __name__ == "__main__":
    main()
