#!/usr/bin/env python3
"""闭环取证（只读诊断脚本；不改任何行为代码）。

在同一冻结评测协议下重跑指定 spec，并记录逐 env-step 状态 + 逐策略步 plan，
以区分"预瞄本身不可跟" vs "tracker 增益不适配" vs "策略动作错误"。

模式
----
- ``lqr``           : 完整复现评测协议（ckpt + LqrTracker），带逐 step 记录；
- ``lqr_gain``      : 同上，但用 ``--tracker-json`` 覆盖 LqrTracker 参数（增益/预瞄）；
- ``exact``         : ckpt + ExactTracker（把 plan 精确置位）→ "若 plan 被完美执行会怎样"；
- ``baseline``      : PurePursuitIDMPolicy（规则专家参照");
- ``arc``           : 合成圆弧参考实验：每 0.5 s 以当前位姿为原点设 κ 参考
                      （``--arc-radius`` / ``--arc-speed``），测 LQR 的曲率-速度可行域。

用法::

    tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr --ids 0,5,11,14,18,19,22,23,28,29,30,32,34,40,43,44,47 --out runs/forensics/closed_lqr.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CKPT = "runs/train/il_v2_10x10_b_fixed/stage_b/final.pt"
SPEC = "env/specs/scenarios_val_slice50.json"


def _wrap(o: object) -> object:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in np.asarray(o, dtype=np.float64).reshape(-1).tolist()]
    return o


def _lane_block(env) -> str:
    try:
        from env.scenario.behaviors import map_info

        info = map_info(env)
        idx = getattr(env.agent, "lane_index", None)
        if info is not None and idx is not None:
            return str(info.lane_block.get(tuple(idx), "?"))
    except Exception:  # noqa: BLE001
        return "?"
    return "?"


def _lane_lat(env) -> float:
    """ego 相对当前车道中心线的横向偏差（m；正值 = 车道中心左侧）。"""
    try:
        from env.scenario.behaviors import lane_projection

        lane = getattr(env.agent, "lane", None)
        if lane is None:
            return float("nan")
        proj = lane_projection(lane, env.agent.position)
        if proj is None:
            return float("nan")
        return float(proj[1]) if not isinstance(proj, float) else float(proj)
    except Exception:  # noqa: BLE001
        return float("nan")


def _lane_curvature(env) -> float:
    """自车当前车道、当前位置的曲率（1/m，finite difference）。"""
    try:
        from env.scenario.behaviors import lane_curvature, lane_projection

        lane = getattr(env.agent, "lane", None)
        if lane is None:
            return float("nan")
        proj = lane_projection(lane, env.agent.position)
        if proj is None:
            return float("nan")
        return float(lane_curvature(lane, proj[0]))
    except Exception:  # noqa: BLE001
        return float("nan")


def _nav_checkpoints(env):
    try:
        nav = getattr(env.agent, "navigation", None)
        if nav is None:
            return []
        return [[float(p[0]), float(p[1])] for p in nav.get_checkpoints()]
    except Exception:  # noqa: BLE001
        return []


class InstrumentedCkpt:
    """评测协议（pipeline.eval_runner._CkptController）的带记录副本。

    与 ``_CkptController`` 的差异**仅**为：记录 plan/动作/误差/位姿，并允许注册
    带自定义参数的 LqrTracker（``tracker_json``）；执行语义逐行照抄。
    """

    def __init__(self, env, spec, task, *, tracker_json: dict | None = None, seed: int = 0, dtheta_gain: float = 1.0):
        import torch

        from pipeline import eval_runner as ev

        self._ev = ev
        self.spec = spec
        self.seed = int(seed)
        self.dtheta_gain = float(dtheta_gain)
        self.records: list = []
        self.steps: list = []
        self.device = torch.device(str(task.get("device") or "cpu"))
        self.model = ev._load_ckpt_model(str(task["ckpt"]), task.get("model_config") or {}, str(self.device))
        self.obs_config = dict(task.get("obs_config") or {})
        self.tracker_kind = str(task.get("tracker") or "lqr").lower()
        self.tracker_json = dict(tracker_json or {})
        self.builder = None
        self.tracker = None
        self._steps = 0
        self._action = [0.0, 0.0]
        self._pose_history: list = []
        self.decision_interval = 5

    # ---------------------------------------------------------------- bind
    def bind(self, env) -> None:
        from env.obs.builder import ObservationBuilder

        self.builder = ObservationBuilder(self.obs_config)
        self.builder.reset()
        self._steps = 0
        self._action = [0.0, 0.0]
        self._pose_history = []
        env.prev_policy_action = np.zeros(2, dtype=np.float64)
        if self.tracker_kind == "exact":
            from env.tracking import ExactTracker

            self.tracker = ExactTracker(dt=0.5, hz=10)
            return
        if self.tracker_kind == "arc":
            from env.tracking import LqrTracker

            self.tracker = env.engine.add_policy(
                env.agent.id, LqrTracker, env.agent, self.seed, **self.tracker_json
            )
            if hasattr(self.tracker, "reset"):
                self.tracker.reset()
            return
        from env.tracking import LqrTracker

        self.tracker = env.engine.add_policy(
            env.agent.id, LqrTracker, env.agent, self.seed, **self.tracker_json
        )
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()

    # ---------------------------------------------------------------- 动作
    def _tensors(self, obs):
        import torch

        from pipeline.trainer import NON_OBS_KEYS, squeeze_single_slot

        batch = {k: np.asarray(v, dtype=np.float32)[None] for k, v in obs.items() if k not in NON_OBS_KEYS}
        squeeze_single_slot(batch)
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    def _measured_prev_action(self, env):
        from env.obs.base import wrap_to_pi

        ego = env.agent
        self._pose_history.append((float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)))
        if len(self._pose_history) <= self.decision_interval:
            return None
        window = self._pose_history[-(self.decision_interval + 1):]
        length = 0.0
        for (ax, ay, _), (bx, by, _) in zip(window[:-1], window[1:]):
            length += math.hypot(bx - ax, by - ay)
        return np.asarray([length, float(wrap_to_pi(window[-1][2] - window[0][2]))], dtype=np.float64)

    def action(self, env):
        import torch

        ego = env.agent
        measured = self._measured_prev_action(env)
        if measured is not None:
            env.prev_policy_action = measured
        obs = self.builder.build(env, self.spec)
        rec = {
            "step": int(self._steps),
            "decision": bool(self._steps % self.decision_interval == 0),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": float(ego.heading_theta),
            "speed": float(ego.speed),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "nav_cps": _nav_checkpoints(env),
        }
        if self._steps % self.decision_interval == 0:
            with torch.no_grad():
                output = self.model(self._tensors(obs), rollout=True, world_model=False)
            mu = np.asarray(output["action_mu"].detach().cpu(), dtype=np.float64).reshape(-1)
            plan = np.asarray(output["plan"].detach().cpu(), dtype=np.float64).reshape(-1, 2)
            plan[0] = mu[:2]
            if self.dtheta_gain != 1.0:
                # 纯诊断：只放大"参考"的转向通道，不改模型（判"幅度不足" vs "结构缺陷"）
                plan = np.array(plan, dtype=np.float64, copy=True)
                plan[:, 1] *= self.dtheta_gain
            rec["mu"] = [float(mu[0]), float(mu[1])]
            rec["plan"] = [[float(a), float(b)] for a, b in plan]
            if self.tracker_kind == "exact":
                self.tracker.arm(env, actions=plan)
            else:
                self.tracker.set_reference(plan)
        self._action = [0.0, 0.0]
        self._steps += 1
        rec["_ego"] = ego
        self.records.append(rec)
        self.current = rec
        return self._action

    def post_step(self, env) -> None:
        if self.tracker_kind == "exact" and self.tracker is not None:
            self.tracker.apply(env)

    def action_info(self, env):
        return {"steer": float(self._action[0]), "throttle": float(self._action[1]), "lead_gap_m": -1.0}

    # ---------------------------------------------------------------- tracker 反馈
    def tracker_info(self):
        info = getattr(self.tracker, "action_info", None) or {}
        action = info.get("action") or [None, None]
        return {
            "lqr_error_y": info.get("lqr_error_y"),
            "lqr_error_psi": info.get("lqr_error_psi"),
            "lqr_preview_index": info.get("lqr_preview_index"),
            "lqr_lookahead_m": info.get("lqr_lookahead_m"),
            "lqr_ref_speed_mps": info.get("lqr_ref_speed_mps"),
            "lqr_status": info.get("lqr_status"),
            "steer_applied": action[0],
            "throttle_applied": action[1],
        }


class ArcController(InstrumentedCkpt):
    """合成圆弧参考：每策略步以当前位姿重置 κ 参考（不依赖策略网络）。

    参考 = 6 个 (ds=arc_speed*0.5, dtheta=ds*κ)；κ = 1/radius（左正）。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, radius=8.0, arc_speed=5.0):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.radius = float(radius)
        self.arc_speed = float(arc_speed)

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = {
            "step": int(self._steps),
            "decision": bool(self._steps % self.decision_interval == 0),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": float(ego.heading_theta),
            "speed": float(ego.speed),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "nav_cps": _nav_checkpoints(env),
        }
        if self._steps % self.decision_interval == 0:
            ds = self.arc_speed * 0.5
            dth = ds / self.radius
            plan = np.asarray([[ds, dth]] * 6, dtype=np.float64)
            rec["mu"] = [float(ds), float(dth)]
            rec["plan"] = [[float(a), float(b)] for a, b in plan]
            self.tracker.set_reference(plan)
        self._steps += 1
        rec["_ego"] = ego
        self.records.append(rec)
        self.current = rec
        return [0.0, 0.0]


class InstrumentedBaseline:
    """``_BaselineController``（PurePursuitIDM 规则专家）的带记录适配器（语义全部委托）。"""

    def __init__(self, env, spec, params=None):
        from pipeline import eval_runner as ev

        self._inner = ev._BaselineController(env, spec, params or {})
        self.records: list = []
        self.steps: list = []
        self._steps = 0

    def bind(self, env) -> None:
        self._inner.bind(env)

    def action(self, env):
        ego = env.agent
        rec = {
            "step": int(self._steps),
            "decision": True,  # baseline 每 env step 都决策
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": float(ego.heading_theta),
            "speed": float(ego.speed),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "nav_cps": _nav_checkpoints(env),
        }
        self._steps += 1
        self.records.append(rec)
        return self._inner.action(env)

    def action_info(self, env):
        return self._inner.action_info(env)

    def tracker_info(self):
        return {}

    def params(self):
        return self._inner.params()


class LanePlanController(InstrumentedCkpt):
    """Oracle plan：每 0.5 s 用**车道中心线**构造 3 s 参考（不依赖策略网络），交给同一个 LQR。

    参考构造：从自车当前车道位置沿路由（ego lane → nav.next_ref_lanes）按 0.1 s 间隔采样
    世界系中心线位姿 → 转到自车系 → 按 0.5 s 窗口折算 ``(ds, dtheta)``（与
    ``env.tracking.roundtrip_error`` 同口径），正好 6 段。
    速度：``speed_mode=hold`` 保持当前速度；``limit`` 用车道限速（clip 到 [1, speed_cap]）。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, speed_mode="limit", speed_cap=8.0):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.speed_mode = str(speed_mode)
        self.speed_cap = float(speed_cap)

    def _route_poses(self, env, horizon_s=3.0, dt=0.1):
        ego = env.agent
        nav = getattr(ego, "navigation", None)
        lane = getattr(ego, "lane", None)
        candidates = []
        try:
            candidates = [item for item in list(getattr(nav, "next_ref_lanes", []) or []) if item is not lane]
        except Exception:  # noqa: BLE001
            candidates = []
        lanes = [ln for ln in [lane] if ln is not None]
        # 贪心接续：优先 index 前缀匹配 + 车道号相同，其次起点最近
        while lanes:
            cur = lanes[-1]
            cur_idx = tuple(getattr(cur, "index", ()) or ())
            best, best_key = None, None
            try:
                cur_end = cur.position(float(cur.length), 0.0)
            except Exception:  # noqa: BLE001
                cur_end = None
            for cand in candidates:
                if cand in lanes:
                    continue
                cidx = tuple(getattr(cand, "index", ()) or ())
                if len(cur_idx) < 3 or len(cidx) < 3:
                    continue
                # index = (from_block, block, lane_no)：接续 = 本车道 block == 候选 from_block
                if cidx[0] != cur_idx[1]:
                    continue
                same_lane_no = bool(cidx[2] == cur_idx[2])
                dist = 0.0
                if cur_end is not None:
                    try:
                        dist = float(np.linalg.norm(np.asarray(cand.position(0.0, 0.0)) - np.asarray(cur_end)))
                    except Exception:  # noqa: BLE001
                        dist = 0.0
                key = (0 if same_lane_no else 1, round(dist, 2))
                if best_key is None or key < best_key:
                    best, best_key = cand, key
            if best is None:
                break
            lanes.append(best)
            if len(lanes) >= 3:
                break
        if not lanes:
            return None
        speed = float(ego.speed)
        if self.speed_mode == "limit":
            try:
                lim = float(lane.speed_limit) if lane is not None else float("nan")
                if not (0.0 < lim < 1000.0):
                    lim = 8.0
            except Exception:  # noqa: BLE001
                lim = 8.0
            speed = float(min(max(lim, 1.0), self.speed_cap))
        else:
            speed = float(max(min(speed, self.speed_cap), 1.0))
        # 自车在首条车道的纵向位置
        s = 0.0
        try:
            proj = lane.local_coordinates(ego.position)
            s = float(proj[0])
        except Exception:  # noqa: BLE001
            s = 0.0
        poses = []
        t = dt
        while t <= horizon_s + 1e-9:
            target = t * speed
            remain = target
            cur_s, k = s, 0
            while k < len(lanes):
                ln = lanes[k]
                ln_len = float(getattr(ln, "length", 0.0))
                if cur_s + remain <= ln_len:
                    sample_s = cur_s + remain
                    break
                remain -= max(ln_len - cur_s, 0.0)
                k += 1
                cur_s = 0.0
            else:
                k = len(lanes) - 1
                sample_s = float(getattr(lanes[k], "length", 0.0))
            ln = lanes[k]
            try:
                pos = ln.position(sample_s, 0.0)
                th = float(ln.heading_theta_at(sample_s))
                poses.append([float(pos[0]), float(pos[1]), th])
            except Exception:  # noqa: BLE001
                break
            t += dt
        return np.asarray(poses, dtype=np.float64) if poses else None

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = {
            "step": int(self._steps),
            "decision": bool(self._steps % self.decision_interval == 0),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": float(ego.heading_theta),
            "speed": float(ego.speed),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "nav_cps": _nav_checkpoints(env),
        }
        if self._steps % self.decision_interval == 0:
            poses = self._route_poses(env, horizon_s=3.0, dt=0.1)
            if poses is not None and len(poses) >= 30:
                from env.tracking import world_to_ego
                from env.obs.base import wrap_to_pi

                base = np.array([float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)])
                local = world_to_ego(poses, base)
                plan = []
                for seg in range(6):
                    win = local[seg * 5: (seg + 1) * 5 + 1]
                    ds = float(np.linalg.norm(np.diff(win[:, :2], axis=0), axis=1).sum())
                    dth = float(wrap_to_pi(win[-1, 2] - win[0, 2]))
                    plan.append([ds, dth])
                plan = np.asarray(plan, dtype=np.float64)
                rec["mu"] = [float(plan[0, 0]), float(plan[0, 1])]
                rec["plan"] = [[float(a), float(b)] for a, b in plan]
                self.tracker.set_reference(plan)
        self._steps += 1
        rec["_ego"] = ego
        self.records.append(rec)
        self.current = rec
        return [0.0, 0.0]


class OracleReplayController(InstrumentedCkpt):
    """Oracle 路径执行：参考 = 同一 spec 上规则专家（PurePursuitIDM）实测轨迹的**未来 3 s**。

    用于判定"执行栈（LqrTracker）能否跟上一条**好路径**"：把专家轨迹按当前位姿转到自车系，
    每 0.5 s 刷新一次参考。速度参考同样来自专家轨迹弦长。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, oracle_poses=None):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.oracle = np.asarray(oracle_poses, dtype=np.float64)  # (N,3) 世界系
        self.k = 0

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = {
            "step": int(self._steps),
            "decision": bool(self._steps % self.decision_interval == 0),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": float(ego.heading_theta),
            "speed": float(ego.speed),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "nav_cps": _nav_checkpoints(env),
        }
        if self._steps % self.decision_interval == 0 and self.oracle is not None and len(self.oracle) > 1:
            from env.tracking import world_to_ego

            base = np.array([float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)])
            start = min(self.k, len(self.oracle) - 2)
            # 用最近点重定位 cursor（对位姿漂移鲁棒），再取未来 30 点
            window = self.oracle[max(0, start - 20): start + 60]
            if len(window):
                d = np.linalg.norm(window[:, :2] - base[:2], axis=1)
                start = max(0, start - 20) + int(np.argmin(d))
            chunk = self.oracle[start: start + 31]
            if len(chunk) >= 2:
                local = world_to_ego(chunk, base)
                rec["plan"] = [[float(p[0]), float(p[1]), float(p[2])] for p in local]
                rec["mu"] = [float(local[0, 0]), float(local[0, 2])]
                self.tracker.set_reference(local)  # (N,3) 自车系位姿参考
            self.k = start + 5
        self._steps += 1
        rec["_ego"] = ego
        self.records.append(rec)
        self.current = rec
        return [0.0, 0.0]


def run_episode(env, spec, controller, *, max_steps: int = 1000):
    """按 _run_episode 的执行顺序跑一条 episode，并给 controller 的记录补 env 侧信息。"""
    from pipeline import eval_runner as ev

    reset_out = env.reset()
    controller.bind(env)
    ego = env.agent
    info: dict = {}
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        info = dict(reset_out[1])
    steps = 0
    for step_index in range(max_steps):
        action = controller.action(env)
        _, _, terminated, truncated, info = env.step(action)
        if hasattr(controller, "post_step"):
            controller.post_step(env)
        info = info if isinstance(info, dict) else {}
        steps = step_index + 1
        trk = controller.tracker_info()
        row = {
            "step": int(step_index),
            "t": round(0.1 * steps, 2),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": round(float(ego.heading_theta), 5),
            "speed": round(float(info.get("velocity", float(ego.speed))), 3),
            "rc": round(float(info.get("route_completion", 0.0)), 4),
            "out_of_road": bool(info.get("out_of_road", False)),
            "crash": bool(ev._is_crash(info)),
            "arrive": bool(info.get("arrive_dest", False)),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "steer": float(action[0]),
            "throttle": float(action[1]),
        }
        row.update({k: _wrap(v) for k, v in trk.items()})
        if controller.records:
            prev = controller.records[-1]
            prev["env_after"] = row
            prev["steer"] = row.get("steer")
        controller.steps.append(row)
        if terminated or truncated:
            break
    last = controller.steps[-1] if controller.steps else {}
    return {
        "steps": steps,
        "termination": ("arrive_dest" if last.get("arrive") else "collision" if last.get("crash")
                        else "out_of_road" if last.get("out_of_road") else "max_step"),
        "rc_final": last.get("rc"),
        "rc_max": max([s.get("rc") or 0.0 for s in controller.steps] or [0.0]),
        "speed_mean": float(np.mean([s["speed"] for s in controller.steps])) if controller.steps else float("nan"),
    }


def _build_env(spec, config):
    from pipeline import eval_runner as ev
    from env.metadrive_env import build_env

    return ev._unwrap_env(build_env(spec, traffic_density=None, use_render=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="forensics_closed_loop.py")
    parser.add_argument("--ids", default="", help="逗号分隔 spec id（缺省=全部焦点几何）")
    parser.add_argument("--mode", default="lqr", choices=("lqr", "lqr_gain", "exact", "baseline", "arc", "laneplan", "oracle"))
    parser.add_argument("--speed-mode", default="limit", choices=("hold", "limit"))
    parser.add_argument("--dtheta-gain", type=float, default=1.0, help="诊断：放大 plan 的转向通道")
    parser.add_argument("--speed-cap", type=float, default=8.0)
    parser.add_argument("--spec", default=SPEC)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--tracker-json", default="{}")
    parser.add_argument("--arc-radius", type=float, default=8.0)
    parser.add_argument("--arc-speed", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from pipeline import eval_runner as ev
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    config = ev.load_config(args.config)
    env_cfg = dict(config.get("env") or {})
    task = {
        "policy": "ckpt",
        "ckpt": args.ckpt,
        "model_config": {
            "hidden_dim": config.get("hidden_dim", 128),
            "moe": dict(config.get("moe") or {}),
            "world_model": dict(config.get("world_model") or {}),
        },
        "obs_config": dict(env_cfg.get("obs") or {}),
        "device": args.device,
        "tracker": "exact" if args.mode == "exact" else "lqr",
    }
    tracker_json = json.loads(args.tracker_json)

    from env.scenario.spec import load_specs

    specs = list(load_specs(args.spec))
    wanted = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids.strip() else None
    if wanted is None:
        focus = {"curve", "roundabout", "uturn", "tollgate"}
        wanted = [int(s.id) for s in specs if str(getattr(s, "labels", {}).get("geometry", "")) in focus]
    chosen = [s for s in specs if int(s.id) in set(wanted)]

    report = {"mode": args.mode, "tracker_json": tracker_json, "episodes": []}
    started = time.time()
    for spec in chosen:
        t0 = time.time()
        try:
            if args.mode == "oracle":
                # pass 1：规则专家跑一遍，取实测轨迹作为 oracle 路径
                env1 = _build_env(spec, config)
                base_ctrl = InstrumentedBaseline(env1, spec, {})
                run_episode(env1, spec, base_ctrl, max_steps=args.max_steps)
                oracle = [[s["x"], s["y"], s["theta"]] for s in base_ctrl.steps]
                try:
                    env1.close()
                except BaseException:
                    pass
                env = _build_env(spec, config)
                controller = OracleReplayController(env, spec, task, tracker_json=tracker_json,
                                                    oracle_poses=oracle)
            else:
                env = _build_env(spec, config)
                if args.mode == "baseline":
                    controller = InstrumentedBaseline(env, spec, {})
                elif args.mode == "arc":
                    controller = ArcController(env, spec, task, tracker_json=tracker_json,
                                               radius=args.arc_radius, arc_speed=args.arc_speed)
                elif args.mode == "laneplan":
                    controller = LanePlanController(env, spec, task, tracker_json=tracker_json,
                                                    speed_mode=args.speed_mode, speed_cap=args.speed_cap)
                else:
                    controller = InstrumentedCkpt(env, spec, task, tracker_json=tracker_json,
                                                   dtheta_gain=args.dtheta_gain)
            outcome = run_episode(env, spec, controller, max_steps=args.max_steps)
        except BaseException as exc:  # noqa: BLE001
            import traceback

            outcome = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        finally:
            try:
                env.close()  # type: ignore[has-type]
            except BaseException:
                pass
        entry = {
            "id": int(spec.id),
            "geometry": str(getattr(spec, "labels", {}).get("geometry", "?")),
            "geometry_seq": list(getattr(spec, "geometry", []) or []),
            "difficulty": str(getattr(spec, "difficulty", "?")),
            "outcome": outcome,
            "wall_s": round(time.time() - t0, 1),
        }
        records = getattr(controller, "records", None)
        if records is not None:
            entry["records"] = records
            entry["steps"] = getattr(controller, "steps", [])
        report["episodes"].append(entry)
        print(
            f"[forensics] id={spec.id} {entry['geometry']:<12} "
            f"term={outcome.get('termination')} rc={outcome.get('rc_final')} steps={outcome.get('steps')} "
            f"({entry['wall_s']}s)",
            flush=True,
        )
    report["elapsed_s"] = round(time.time() - started, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[forensics] → {out_path} ({report['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
