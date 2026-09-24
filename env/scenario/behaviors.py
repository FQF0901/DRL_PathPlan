"""脚本化 cut-in / cut-out 行为（确定性、可复现）。

用途与关键接口
- ``install(env, spec)``：在 ``env`` 上挂载 ``ScenarioBehaviorManager``（引擎已就绪且有地图则立即生成
  脚本车；否则延迟到首次 ``reset()``），并把事件窗口与状态写进 ``env._scenario_events``。
- ``event_state(env)``：返回当前事件窗口 + 脚本车**逐步可观测几何**（距离 / 横向偏移 / 纵向相对位置），
  供 ``labels.compute_step_labels`` 计算标签；没有安装行为时返回空状态（窗口恒 inactive）。
- ``map_info(env)``：按 block 结构缓存的地图信息（lane -> block、路口/环岛/汇入区采样点）。
- ``spec_field(spec, name, default)``：兼容 dataclass 与 dict 两种 spec 形态。

为什么这样实现（0.4.3 源码核对，行号对应安装包 ``.venv/.../metadrive``）
- **安装版没有 `WaypointPolicy`**：``metadrive/policy/`` 下只有
  ``base_policy / env_input_policy / idm_policy / expert_policy / lange_change_policy /
  manual_control_policy / replay_policy / AI_protect_policy``；不存在
  ``metadrive.policy.waypoint_policy``（契约名字来自更新的版本/文档）。因此本模块用
  ``WaypointPolicy(BasePolicy)`` 实现等价回放，并沿用**已验证**的
  ``ReplayTrafficParticipantPolicy.act`` 模式：直接写 position/heading/velocity、返回 None，
  以免 ``BaseVehicle.before_step`` 覆盖控制量（replay_policy.py:41-75）。
- 脚本车经 ``BaseEngine.spawn_object`` 生成（base_engine.py:126-164），策略经
  ``BaseEngine.add_policy`` 注册（base_engine.py:98-104）；管理器的
  ``before_step / after_step / reset / after_reset`` 由 ``BaseEngine`` 按 ``PRIORITY`` 调用
  （base_engine.py:349-382 reset 三段, 416-429 before_step, 462-473 after_step）。
  ``PRIORITY=11`` 保证 reset/after_reset 在 agent_manager(10) 之后、before_step 在物理步之前。
- **横向符号**：``StraightLane.direction_lateral = [dy, -dx]``（straight_lane.py:46,61），
  即 **正 lateral 在车道方向的右侧**；与 ``BaseVehicle._dist_to_route_left_right``
  （base_vehicle.py:517-526）一致 -> 左邻车道用 ``-width``，右邻用 ``+width``。
- **车道 id 从左到右递增**：``CreateRoadFrom`` 的 ``toward_smaller_lane_index`` 分支
  （create_pg_block_utils.py:68-186）把最左车道放在 id 0；``PGBlock`` 取 ``positive_lanes[-1]``
  作为 "most right or outside lane"（pg_block.py:121）-> 左邻 ``id-1``、右邻 ``id+1``。
- **时间**：``physics_world_step_size=0.02 x decision_repeat=5`` -> 每个 ``env.step`` = 0.1 s
  （envs/base_env.py:188-190, 466）。
- 车辆 spawn 需要 ``navigation_module``（``BaseVehicle.lane`` 依赖 ``navigation.current_lane``，
  base_vehicle.py:1000-1006）。

事件语义（与 L1a spec 对齐）
- ``spec.traffic["events"][*]`` 字段：``type``(cut_in/cut_out) / ``side`` / ``trigger_step`` /
  ``duration_steps`` / ``gap_m`` / ``speed_mps``（步数按 10 Hz env step；速度 m/s）。
- 窗口 = ``[trigger_step, trigger_step + duration_steps)``；``gap_m`` 是机动开始时脚本车相对自车的
  纵向间距（cut-in 从侧车道切进自车前方的这个间隙，cut-out 从自车前方这个间距处离开）。
- 机动前脚本车在侧邻车道（cut-out 在本车道）以 ``speed_mps`` 行驶；**纵向相对位置逐帧对自车实时
  车道重锚定**（``rel = actor_long - ego_long``，按二者速度差积分），因此跨 block/换道都跟随自车路线，
  不需要事先知道地图拓扑。
- 机动 + 0.5 s 后把车交给 ``IDMPolicy`` 继续作为普通交通行驶（避免"停车/瞬移/凭空消失"式时间悖论）；
  提交前用同一 ``BasePolicy`` 回放，位姿逐帧连续。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils.math import wrap_to_pi

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------

STEP_DT = 0.1  # 秒 / env.step（见模块 docstring 的 physics 参数核对）

MANAGER_NAME = "scenario_behavior_manager"
MANAGER_PRIORITY = 11  # > agent_manager/traffic_manager(10)，< replay/record；base_manager.py:14

EVENT_KINDS = ("cut_in", "cut_out")

# 事件窗口缺省值（spec 未给 trigger/duration 时按 seed 确定性合成，保证 validator 可见）
DEFAULT_TRIGGER_BASE = 10  # 1.0 s
DEFAULT_TRIGGER_SPAN = 10
DEFAULT_DURATION_STEPS = 40  # 4.0 s

# 轨迹缺省值
DEFAULT_GAP_M = 15.0  # 机动开始时脚本车相对自车的纵向间距
DEFAULT_MERGE_S = 1.6  # 横向并线时长
MAX_MERGE_S = 2.0
MIN_MERGE_S = 0.8
POST_WINDOW_S = 0.5  # 窗口结束后多久交给 IDM
MIN_REL_M = 3.0  # 脚本车纵向相对位置下界（防与自车重叠）
MAX_REL_M = 60.0  # 上界（防止相对位置无限漂移）
HANDOVER_PAD_STEPS = int(POST_WINDOW_S / STEP_DT)

# spawn 位姿避让（确定性：发现占用就沿纵向平移固定步长）
CLEARANCE_M = 6.0
CLEARANCE_SHIFT_M = 8.0
CLEARANCE_TRIES = 3

INTERSECTION_BLOCK_IDS = ("X", "T", "U")  # intersection.py / t_intersection.py
ROUNDABOUT_BLOCK_IDS = ("O",)  # roundabout.py
MERGE_BLOCK_IDS = ("r", "R", "y", "Y")  # 匝道 + 瓶颈 merge/split

MAP_SAMPLE_INTERVAL = 3.0  # 区域采样间隔（m），用于 near_* 距离


# --------------------------------------------------------------------------------------
# spec 访问（兼容 ScenarioSpec dataclass 与 dict）
# --------------------------------------------------------------------------------------


def spec_field(spec: Any, name: str, default: Any = None) -> Any:
    """读取 spec 字段：优先属性，其次 dict key；缺失返回 default。"""
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _traffic_dict(spec: Any) -> dict:
    traffic = spec_field(spec, "traffic", {}) or {}
    return traffic if isinstance(traffic, dict) else {}


def _traffic_patterns(spec: Any) -> list[str]:
    return [str(p).lower() for p in (_traffic_dict(spec).get("patterns") or ())]


# --------------------------------------------------------------------------------------
# 几何工具（observable 语义的公共实现）
# --------------------------------------------------------------------------------------


def lane_point(lane: Any, s: float, lateral: float) -> tuple[np.ndarray, float]:
    """车道坐标 (s, lateral) -> (世界坐标, 航向)。

    超出车道范围时按端点切线外推（CircularLane.position 外推会绕圆心，故不可直接使用）。
    """
    length = float(lane.length)
    if s < 0.0:
        base_s, delta = 0.0, float(s)
    elif s > length:
        base_s, delta = length, float(s - length)
    else:
        base_s, delta = float(s), 0.0
    point = np.asarray(lane.position(base_s, lateral), dtype=float)
    heading = float(lane.heading_theta_at(base_s))
    if abs(delta) > 1e-9:
        point = point + delta * np.array([math.cos(heading), math.sin(heading)], dtype=float)
    return point, heading


def lane_curvature(lane: Any, s: float, ds: float = 5.0) -> float:
    """由车道 heading 的有限差分估计曲率 [1/m]（对 Straight/Circular/PointLane 均可用）。"""
    length = float(lane.length)
    s0 = float(np.clip(s - ds, 0.0, length))
    s1 = float(np.clip(s + ds, 0.0, length))
    if s1 - s0 < 1.0:
        return 0.0
    dh = wrap_to_pi(lane.heading_theta_at(s1) - lane.heading_theta_at(s0))
    return float(abs(dh) / (s1 - s0))


def lane_projection(lane: Any, position: np.ndarray) -> Optional[tuple[float, float]]:
    """世界坐标 -> (longitudinal, lateral)；失败返回 None（lane 缺失/退化几何）。"""
    if lane is None:
        return None
    try:
        long, lat = lane.local_coordinates(position)
        return float(long), float(lat)
    except Exception:  # noqa: BLE001 - 容错：标签/校验不应因单帧几何失败而崩
        return None


def _sample_centerline(lane: Any) -> np.ndarray:
    """采样车道中心线点集 (N,2)，用于 near_* 距离。"""
    length = float(lane.length)
    longs = np.arange(0.0, max(length, MAP_SAMPLE_INTERVAL), MAP_SAMPLE_INTERVAL)
    points = [lane_point(lane, float(min(s, length)), 0.0)[0] for s in longs]
    return np.asarray(points, dtype=float)


# --------------------------------------------------------------------------------------
# 地图信息（lane -> block；路口/环岛/汇入区采样点）
# --------------------------------------------------------------------------------------


@dataclass
class MapInfo:
    """按当前地图缓存的场景几何信息（地图重建后自然失效，见 ``map_info``）。"""

    lane_block: dict[tuple, str] = field(default_factory=dict)
    block_ids: list[str] = field(default_factory=list)
    intersection_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    roundabout_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    merge_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def _build_map_info(map_: Any) -> MapInfo:
    info = MapInfo()
    for block in list(getattr(map_, "blocks", []) or []):
        block_id = str(getattr(block, "ID", "?") or "?")
        info.block_ids.append(block_id)
        network = getattr(block, "block_network", None)
        if network is None or not hasattr(network, "get_all_lanes"):
            continue
        try:
            lanes = list(network.get_all_lanes())
        except Exception:  # noqa: BLE001
            lanes = []
        for lane in lanes:
            index = getattr(lane, "index", None)
            if index is not None:
                info.lane_block[tuple(index)] = block_id
            try:
                points = _sample_centerline(lane)
            except Exception:  # noqa: BLE001
                continue
            if block_id in INTERSECTION_BLOCK_IDS:
                info.intersection_points = np.vstack([info.intersection_points, points])
            elif block_id in ROUNDABOUT_BLOCK_IDS:
                info.roundabout_points = np.vstack([info.roundabout_points, points])
            elif block_id in MERGE_BLOCK_IDS:
                info.merge_points = np.vstack([info.merge_points, points])
    return info


def map_info(env: Any) -> Optional[MapInfo]:
    """获取（并缓存）当前地图的 MapInfo；地图未就绪返回 None。

    缓存挂在 map 对象上：``store_map=False`` 时每次 reset 重建地图 -> 自动失效；
    可选 LRU 复用时地图结构不变，缓存长期有效。
    """
    engine = getattr(env, "engine", env)
    map_ = getattr(engine, "current_map", None) if engine is not None else None
    if map_ is None:
        return None
    cached = getattr(map_, "_scenario_map_info", None)
    if isinstance(cached, MapInfo):
        return cached
    try:
        info = _build_map_info(map_)
        setattr(map_, "_scenario_map_info", info)
    except Exception:  # noqa: BLE001
        return None
    return info


def nearest_distance(points: Optional[np.ndarray], position: np.ndarray) -> Optional[float]:
    """点到采样点集的最小欧氏距离；无点返回 None。"""
    if points is None or len(points) == 0:
        return None
    diff = points - np.asarray(position, dtype=float)[:2]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))


# --------------------------------------------------------------------------------------
# 自车 / 车道访问
# --------------------------------------------------------------------------------------


def ego_vehicle(env: Any) -> Optional[BaseVehicle]:
    """兼容 env 与 engine 两种入参，返回单智能体 ego（无则 None）。"""
    engine = getattr(env, "engine", env)
    if engine is None:
        return None
    manager = getattr(engine, "agent_manager", None)
    if manager is None:
        return None
    for attr in ("active_agents", "episode_created_agents"):
        agents = getattr(manager, attr, None)
        if isinstance(agents, dict) and agents:
            return next(iter(agents.values()))
    return None


def ego_lane(engine: Any, ego: BaseVehicle) -> Any:
    """自车当前车道（navigation 优先，失败则按位置反查最近车道）。"""
    lane = getattr(ego, "lane", None)
    if lane is not None:
        return lane
    roadmap = getattr(getattr(engine, "current_map", None), "road_network", None)
    if roadmap is None:
        return None
    try:
        index = roadmap.get_closest_lane_index(ego.position)
        if isinstance(index, (list, tuple)) and len(index) and isinstance(index[0], tuple):
            index = index[0]
        return roadmap.get_lane(index)
    except Exception:  # noqa: BLE001
        return None


def _ego_lane_index(engine: Any, ego: BaseVehicle, lane: Any) -> Optional[tuple]:
    index = getattr(ego, "lane_index", None)
    if index is None:
        index = getattr(lane, "index", None)
    if index is None:
        return None
    return tuple(index)


def _ratio_of(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------------------
# 事件请求解析（spec -> requests）
# --------------------------------------------------------------------------------------


def _normalize_request(raw: Any, kind: str) -> Optional[dict]:
    """把 spec 里各种写法归一到 §"事件语义" 的字段集合。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        req = dict(raw)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        req = {"trigger_step": int(raw[0]), "duration_steps": int(raw[1])}
    elif isinstance(raw, (list, tuple)) and len(raw) == 1:
        req = {"trigger_step": int(raw[0])}
    else:
        return None
    req["type"] = kind
    return req


def event_requests(spec: Any) -> dict[str, dict]:
    """解析 ``spec.traffic`` 的事件请求，返回 ``{key: request}``（key 唯一，可含 ``#2``）。

    规范形式（L1a generator）::

        traffic["events"] = [
            {"type": "cut_in", "side": "left", "trigger_step": 40,
             "duration_steps": 45, "gap_m": 12.0, "speed_mps": 8.3}, ...]

    兼容简写：``traffic["cut_in"] = {..}`` / ``(trigger, duration)``、``traffic["cut_in_window"]``、
    以及 ``patterns`` 里声明了 cut_in/cut_out 但没有事件参数（按 seed 合成窗口）。
    """
    requests: dict[str, dict] = {}
    counters: dict[str, int] = {kind: 0 for kind in EVENT_KINDS}

    def _store(req: dict) -> None:
        kind = str(req.get("type", "")).lower()
        if kind not in EVENT_KINDS:
            return
        counters[kind] += 1
        key = kind if counters[kind] == 1 else f"{kind}#{counters[kind]}"
        req["key"] = key
        requests[key] = req

    traffic = _traffic_dict(spec)
    for raw in traffic.get("events") or ():
        if isinstance(raw, dict):
            _store({k: v for k, v in raw.items() if k != "key"})
    for kind in EVENT_KINDS:
        if counters[kind] > 0:
            continue
        req = _normalize_request(traffic.get(kind), kind)
        window = traffic.get(f"{kind}_window")
        if req is None and isinstance(window, (list, tuple)) and len(window) >= 2:
            req = {"trigger_step": int(window[0]), "duration_steps": int(window[1]) - int(window[0])}
            req["type"] = kind
        if req is None and kind in _traffic_patterns(spec):
            req = {"type": kind}
        if req is not None:
            _store(req)
    return requests


def _request_window(req: dict, seed: int) -> tuple[int, int]:
    """事件窗口 [start, end)（env.step 计）；缺省按 seed 合成，保证 50-step rollout 内可见。"""
    start = req.get("trigger_step", req.get("start_step", req.get("start")))
    if start is None:
        start = DEFAULT_TRIGGER_BASE + (abs(int(seed)) % DEFAULT_TRIGGER_SPAN)
    start = int(max(int(start), 0))
    duration = req.get("duration_steps", req.get("duration"))
    if duration is None:
        duration = DEFAULT_DURATION_STEPS
    duration = int(max(int(duration), 5))
    return start, start + duration


# --------------------------------------------------------------------------------------
# 轨迹计划（逐帧对自车实时车道重锚定）
# --------------------------------------------------------------------------------------


@dataclass
class ActorPlan:
    """一辆脚本车的确定性回放计划。

    纵向不是绝对坐标，而是**对自车实时车道的相对纵向位置** ``rel``（米，正=自车前方）：
    ``rel`` 按 ``(speed_mps - ego.speed) * dt`` 积分，横向按 ``lateral_at(t)`` 在自车车道系内插值。
    这样跨 block / 自车换道都会自然跟随，无需预知地图拓扑。
    """

    kind: str  # "cut_in" | "cut_out"
    key: str  # 事件键（state["events"] 的 key）
    side: str  # "left" | "right"
    ego_id: str  # 自车 object id（逐帧查找）
    gap_m: float  # t=0 的纵向相对位置（自车前方 gap_m）
    speed: float  # 脚本车速度 m/s
    lat_side: float  # 侧邻车道的横向偏移（自车车道系；正=右）
    t_trigger: float  # 机动开始时刻 [s]（相对 spawn）
    merge_duration: float  # 横向并线时长 [s]
    window_start_step: int
    window_end_step: int
    spawn_lane_index: tuple
    spawn_longitude: float
    base_lane: Any  # spawn 时的自车车道（自车车道不可得时的回退几何）
    degraded: bool = False  # 相邻车道缺失/空间不足等降级
    spawn_step: int = 0
    rel: float = 0.0  # 当前纵向相对位置（逐帧更新）
    last_ego_long: float = 0.0
    initial_position: Optional[np.ndarray] = None  # spawn 位姿（重叠检查用）

    # ---- 时间 ----
    @property
    def handover_step(self) -> int:
        """窗口结束 + POST_WINDOW_S 后交给 IDM（避免停车/消失式悖论）。"""
        return int(self.window_end_step + HANDOVER_PAD_STEPS)

    def lateral_at(self, t: float) -> float:
        """横向偏移（自车车道系）：approach 在侧车道，merge 平滑过渡，之后贴在目标车道。"""
        target = 0.0 if self.kind == "cut_in" else self.lat_side
        if t <= self.t_trigger:
            return self.lat_side if self.kind == "cut_in" else 0.0
        merge_end = self.t_trigger + max(self.merge_duration, 1e-3)
        if t >= merge_end:
            return target
        start = self.lat_side if self.kind == "cut_in" else 0.0
        u = (t - self.t_trigger) / max(self.merge_duration, 1e-3)
        u = min(max(u, 0.0), 1.0)
        return float(start + (target - start) * (u * u * (3.0 - 2.0 * u)))

    def pose(self, lane: Any, ego_long: float, t: float) -> tuple[np.ndarray, float]:
        """返回 t 时刻位姿；``lane`` 为自车当前车道（None 时回退 base_lane）。"""
        if lane is None:
            lane = self.base_lane
        s = ego_long + self.rel
        return lane_point(lane, s, self.lateral_at(t))

    def velocity(self, heading: float) -> np.ndarray:
        return self.speed * np.array([math.cos(heading), math.sin(heading)], dtype=float)

    def vehicle_config(self) -> dict:
        """脚本车 spawn 配置；navigation_module 显式给出以保证 ``.lane/.lane_index`` 可用。"""
        return {
            "navigation_module": NodeNetworkNavigation,
            "spawn_lane_index": tuple(self.spawn_lane_index),
            "spawn_longitude": float(self.spawn_longitude),
            "spawn_lateral": 0.0,
            "show_navi_mark": False,
            "show_dest_mark": False,
            "show_line_to_dest": False,
            "enable_reverse": False,
            "overtake_stat": False,
        }


class WaypointPolicy(BasePolicy):
    """脚本车航点回放策略（kinematic playback，逐帧写状态）。

    0.4.3 无 ``metadrive.policy.waypoint_policy.WaypointPolicy``（已核对 policy/ 目录），
    这里按 ``ReplayTrafficParticipantPolicy.act`` 的已验证模式实现：直接写
    position/heading/velocity 后返回 None（replay_policy.py:53-75），使脚本车运动与计划
    **逐帧一致**且与自车行为无关（确定性要求）。
    """

    DEBUG_MARK_COLOR = (0, 160, 255, 255)

    def __init__(self, control_object, plan: ActorPlan, random_seed: Optional[int] = None):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.plan = plan

    def _ego(self) -> Optional[BaseVehicle]:
        try:
            objects = self.engine.get_objects()
        except Exception:  # noqa: BLE001
            return None
        return objects.get(self.plan.ego_id)

    def act(self, *args, **kwargs):  # noqa: ANN002, ANN003 - 与 BasePolicy.act 签名族保持一致
        plan = self.plan
        step = int(self.episode_step)
        t = max(0.0, (step - int(plan.spawn_step))) * STEP_DT
        ego = self._ego()
        lane = ego_lane(self.engine, ego) if ego is not None else None
        ego_speed = plan.speed
        ego_long = plan.last_ego_long
        if ego is not None:
            projection = lane_projection(lane, np.asarray(ego.position, dtype=float))
            if projection is not None:
                ego_long = projection[0]
            try:
                ego_speed = float(np.hypot(*np.asarray(ego.velocity, dtype=float)[:2]))
            except Exception:  # noqa: BLE001
                ego_speed = plan.speed
            plan.last_ego_long = ego_long
        if step > int(plan.spawn_step):  # spawn 当步不积分，保持 t=0 精确位姿
            plan.rel = float(np.clip(plan.rel + (plan.speed - ego_speed) * STEP_DT, MIN_REL_M, MAX_REL_M))
        position, heading = plan.pose(lane, ego_long, t)
        obj = self.control_object
        obj.set_position(position)
        obj.set_heading_theta(float(heading))
        obj.set_velocity(plan.velocity(heading))
        obj.set_angular_velocity(0.0)
        self.action_info["t"] = t
        return None


# --------------------------------------------------------------------------------------
# 计划构建
# --------------------------------------------------------------------------------------


def _resolve_side(side: str, index: tuple, lanes_on_road: list) -> tuple[str, Optional[tuple], float, bool]:
    """选择相邻车道：id 从左到右递增（见模块 docstring）。

    返回 (side, side_lane_index|None, lateral_sign, degraded)。lateral_sign 为脚本车在
    自车车道系里的横向偏移符号：左 = -1（负 lateral 在左侧），右 = +1。
    """
    side = "right" if side == "right" else "left"
    lane_id = int(index[2] or 0)
    sign = 1.0 if side == "right" else -1.0
    offset = 1 if side == "right" else -1
    if 0 <= lane_id + offset < len(lanes_on_road):
        return side, (index[0], index[1], lane_id + offset), sign, False
    other = "right" if side == "left" else "left"
    other_sign = 1.0 if other == "right" else -1.0
    other_offset = 1 if other == "right" else -1
    if 0 <= lane_id + other_offset < len(lanes_on_road):
        return other, (index[0], index[1], lane_id + other_offset), other_sign, True
    return side, None, sign, True


def _occupied(engine: Any, position: np.ndarray, radius: float) -> bool:
    """生成点半径内是否已有车辆（避免脚本车与随机交通重叠）。"""
    objects = getattr(engine, "get_objects", None)
    if objects is None:
        return False
    for obj in objects().values():
        if not isinstance(obj, BaseVehicle):
            continue
        try:
            other = np.asarray(obj.position, dtype=float)[:2]
        except Exception:  # noqa: BLE001
            continue
        if float(np.linalg.norm(other - np.asarray(position, dtype=float)[:2])) < radius:
            return True
    return False


def _build_plan(
    engine: Any,
    req: dict,
    lane: Any,
    index: tuple,
    lanes_on_road: list,
    ego: BaseVehicle,
    ego_long: float,
    width: float,
    seed: int,
) -> Optional[ActorPlan]:
    """由请求 + 自车当前可观测状态构建一条脚本车计划。"""
    kind = str(req.get("type", "")).lower()
    if kind not in EVENT_KINDS:
        return None
    side, side_lane_index, lateral_sign, degraded = _resolve_side(str(req.get("side", "left")), index, lanes_on_road)
    gap = float(np.clip(_ratio_of(req.get("gap_m"), DEFAULT_GAP_M), 3.0, MAX_REL_M))
    speed = max(_ratio_of(req.get("speed_mps"), 8.0), 1.0)
    start_step, end_step = _request_window(req, seed)
    duration_s = max((end_step - start_step) * STEP_DT, STEP_DT)
    merge_duration = float(np.clip(min(DEFAULT_MERGE_S, duration_s * 0.5), MIN_MERGE_S, MAX_MERGE_S))
    t_trigger = max((start_step - int(engine.episode_step)) * STEP_DT, 0.0)

    lat_side = lateral_sign * width
    if side_lane_index is None:  # 两侧都无相邻车道：半车道偏移，避免驶出路面
        lat_side = lateral_sign * 0.5 * width
        degraded = True
    # 说明：脚本车纵向位置逐帧对自车实时车道重锚定（见 ActorPlan 注释），
    # 单帧 `lane_point` 对车道末端外的直线外推不会累积成"驶出路线"，因此不额外标降级。

    # 生成位姿：侧车道（cut-in）/ 自车车道（cut-out）在 gap 处；被随机交通占用时沿纵向平移
    spawn_index = side_lane_index or (index[0], index[1], int(index[2] or 0))
    spawn_lane = lane
    try:
        spawn_lane = engine.current_map.road_network.get_lane(spawn_index)
    except Exception:  # noqa: BLE001
        spawn_index = (index[0], index[1], int(index[2] or 0))
    lateral0 = lat_side if kind == "cut_in" else 0.0
    rel = gap
    s_max = max(float(spawn_lane.length) - 1.0, 3.0)
    position, _ = lane_point(lane, ego_long + rel, lateral0)
    for _ in range(CLEARANCE_TRIES):
        if not _occupied(engine, position, CLEARANCE_M):
            break
        rel = float(np.clip(rel + CLEARANCE_SHIFT_M, MIN_REL_M, MAX_REL_M))
        position, _ = lane_point(lane, ego_long + rel, lateral0)
    spawn_long = float(np.clip(ego_long + rel, 1.0, s_max))

    return ActorPlan(
        kind=kind,
        key=str(req.get("key", kind)),
        side=side,
        ego_id=ego.name,
        gap_m=gap,
        speed=speed,
        lat_side=lat_side,
        t_trigger=t_trigger,
        merge_duration=merge_duration,
        window_start_step=start_step,
        window_end_step=end_step,
        spawn_lane_index=spawn_index,
        spawn_longitude=spawn_long,
        base_lane=lane,
        degraded=degraded,
        rel=rel,
        last_ego_long=ego_long,
        initial_position=position,
    )


def _build_plans(engine: Any, spec: Any, ego: BaseVehicle) -> dict[str, ActorPlan]:
    """按 spec 请求 + 自车当前可观测状态构建全部脚本车计划。"""
    requests = event_requests(spec)
    if not requests:
        return {}
    lane = ego_lane(engine, ego)
    if lane is None:
        return {}
    index = _ego_lane_index(engine, ego, lane)
    if index is None:
        return {}
    graph = getattr(getattr(engine.current_map, "road_network", None), "graph", {})
    lanes_on_road = list(graph.get(index[0], {}).get(index[1], []))
    projection = lane_projection(lane, np.asarray(ego.position, dtype=float))
    if projection is None:
        return {}
    ego_long = projection[0]
    width = float(getattr(lane, "width", 3.5) or 3.5)
    seed = int(spec_field(spec, "seed", 0) or 0)

    plans: dict[str, ActorPlan] = {}
    for key, req in requests.items():
        plan = _build_plan(engine, {**req, "key": key}, lane, index, lanes_on_road, ego, ego_long, width, seed)
        if plan is not None:
            plans[key] = plan
    return plans


# --------------------------------------------------------------------------------------
# 事件状态（env._scenario_events）
# --------------------------------------------------------------------------------------


def _event_entry(kind: str, key: str) -> dict:
    return {
        "type": kind,
        "key": key,
        "start_step": None,
        "end_step": None,
        "duration_steps": None,
        "active": False,
        "fired": False,
        "phase": "pending",
        "degraded": False,
        "actor_id": None,
        "actor_alive": False,
        "handover_step": None,
        "side": None,
        "gap_m": None,
    }


def _new_state() -> dict:
    return {
        "spec_id": None,
        "seed": None,
        "spawn_step": None,
        "episode_step": 0,
        "degraded": False,
        "min_distance_to_ego": None,
        "planned_events": {},
        "events": {},
        "actors": {},
    }


def _actor_observation(ego: BaseVehicle, lane: Any, vehicle: BaseVehicle) -> dict:
    ego_pos = np.asarray(ego.position, dtype=float)
    veh_pos = np.asarray(vehicle.position, dtype=float)
    distance = float(np.linalg.norm(veh_pos[:2] - ego_pos[:2]))
    velocity = np.asarray(vehicle.velocity, dtype=float)
    projection = lane_projection(lane, veh_pos) if lane is not None else None
    ego_projection = lane_projection(lane, ego_pos) if lane is not None else None
    lateral = projection[1] if projection is not None else None
    long_rel = None
    if projection is not None and ego_projection is not None:
        long_rel = float(projection[0] - ego_projection[0])
    width = float(getattr(lane, "width", 3.5) or 3.5) if lane is not None else None
    return {
        "alive": True,
        "vehicle_id": vehicle.name,
        "position": [float(veh_pos[0]), float(veh_pos[1])],
        "speed_mps": float(np.hypot(*velocity[:2])),
        "distance": distance,
        "lateral": float(lateral) if lateral is not None else None,
        "long_rel": long_rel,
        "lane_width": width,
        "in_ego_lane": bool(lateral is not None and width is not None and abs(lateral) <= 0.5 * width),
        "ahead": bool(long_rel is not None and long_rel >= 0.0),
    }


def _empty_state() -> dict:
    return _new_state()


def _manager(env: Any) -> Optional["ScenarioBehaviorManager"]:
    engine = getattr(env, "engine", env)
    if engine is None:
        return None
    manager = getattr(engine, MANAGER_NAME, None)
    return manager if isinstance(manager, ScenarioBehaviorManager) else None


def _refresh_actors(env: Any, manager: "ScenarioBehaviorManager") -> None:
    """把脚本车的实时可观测几何写进 state（供 labels 直接读）。"""
    engine = getattr(env, "engine", env)
    state = manager.state
    state["episode_step"] = int(getattr(engine, "episode_step", 0))
    ego = ego_vehicle(engine)
    lane = ego_lane(engine, ego) if ego is not None else None
    objects = {}
    if engine is not None:
        try:
            objects = engine.get_objects()
        except Exception:  # noqa: BLE001
            objects = {}
    views: dict[str, dict] = {}
    min_distance = state.get("min_distance_to_ego")
    for key, actor in manager.actors.items():
        event = state["events"].get(key)
        vehicle = objects.get(actor.vehicle_id)
        if vehicle is None:
            if event is not None:
                event["actor_alive"] = False
            continue
        if ego is None:
            continue
        view = _actor_observation(ego, lane, vehicle)
        views[key] = view
        if event is not None:
            event["actor_alive"] = True
            event["degraded"] = bool(actor.plan.degraded)
            event["actor_id"] = actor.vehicle_id
        distance = view["distance"]
        min_distance = distance if min_distance is None else min(min_distance, distance)
    state["actors"] = views
    if min_distance is not None:
        state["min_distance_to_ego"] = float(min_distance)


def event_state(env: Any) -> dict:
    """读取（并刷新）脚本事件状态；未安装行为时返回空状态。

    返回纯 Python 标量/列表结构（可安全 deepcopy / JSON 序列化），绝不包含 MetaDrive 对象。
    """
    manager = _manager(env)
    if manager is None:
        return _empty_state()
    try:
        _refresh_actors(env, manager)
        return copy.deepcopy(manager.state)
    except Exception:  # noqa: BLE001 - 观测失败不应中断训练循环
        return _empty_state()


# --------------------------------------------------------------------------------------
# 管理器
# --------------------------------------------------------------------------------------


@dataclass
class _Actor:
    key: str
    kind: str
    vehicle_id: str
    plan: ActorPlan
    mode: str = "script"  # "script"：WaypointPolicy 回放；"idm"：机动结束后交还 IDM


class ScenarioBehaviorManager(BaseManager):
    """按 spec 生成 / 驱动 / 清理脚本 cut-in/cut-out 车辆。

    PRIORITY=11：reset/after_reset 在 agent_manager(10) 之后 -> ego 已存在；
    before_step 在 traffic_manager 之后、物理步之前 -> 本步位姿准时生效。
    """

    PRIORITY = MANAGER_PRIORITY

    def __init__(self):
        super().__init__()
        self._spec: Any = None
        self._requests: dict[str, dict] = {}
        self._actors: dict[str, _Actor] = {}
        self._installed_epoch: Optional[int] = None
        self.state: dict = _new_state()

    # ---- 对外状态访问 ----
    @property
    def actors(self) -> dict[str, _Actor]:
        return self._actors

    @property
    def installed_epoch(self) -> Optional[int]:
        return self._installed_epoch

    # ---- 安装 ----
    def install(self, spec: Any, spawn_now: bool = False) -> None:
        """登记 spec 并重置内部状态；``spawn_now`` 时若地图就绪立即生成。"""
        self._clear_actors()
        self._spec = spec
        self._requests = event_requests(spec)
        self._installed_epoch = None
        self._reset_state()
        if spawn_now:
            self.spawn_actors_if_needed()

    def _reset_state(self) -> None:
        """原地重置 state（保持 env._scenario_events 的别名有效），并预登记计划窗口。"""
        seed = int(spec_field(self._spec, "seed", 0) or 0)
        fresh = _new_state()
        self.state.clear()
        self.state.update(fresh)
        self.state["spec_id"] = spec_field(self._spec, "id")
        self.state["seed"] = spec_field(self._spec, "seed")
        for key, req in self._requests.items():
            start_step, end_step = _request_window(req, seed)
            kind = str(req.get("type", ""))
            entry = _event_entry(kind, key)
            entry.update(
                start_step=start_step,
                end_step=end_step,
                duration_steps=end_step - start_step,
                side=str(req.get("side", "")) or None,
                gap_m=float(_ratio_of(req.get("gap_m"), DEFAULT_GAP_M)),
            )
            self.state["events"][key] = entry
            self.state["planned_events"][key] = {
                "type": kind,
                "side": str(req.get("side", "")) or None,
                "trigger_step": start_step,
                "duration_steps": end_step - start_step,
                "gap_m": float(_ratio_of(req.get("gap_m"), DEFAULT_GAP_M)),
                "speed_mps": float(_ratio_of(req.get("speed_mps"), 8.0)),
            }

    # ---- 生命周期（由 BaseEngine 调用）----
    def reset(self) -> dict:
        return dict()

    def after_reset(self) -> dict:
        self.spawn_actors_if_needed()
        self._update_windows(int(self.engine.episode_step))
        _refresh_actors(self.engine, self)
        return dict()

    def before_reset(self) -> None:
        super().before_reset()  # 清理 spawned_objects（引擎会在 _object_clean_check 前调用）
        self._actors = {}
        self._installed_epoch = None
        self._reset_state()

    def before_step(self, *args, **kwargs) -> dict:
        if self._spec is None:
            return dict()
        self.spawn_actors_if_needed()
        if not self._actors:
            return dict()
        step = int(self.engine.episode_step)
        objects = self.engine.get_objects()
        for actor in list(self._actors.values()):
            vehicle = objects.get(actor.vehicle_id)
            policy = self.engine.get_policy(actor.vehicle_id)
            if vehicle is None or policy is None:
                continue
            action = policy.act()
            if action is not None:  # WaypointPolicy 返回 None（状态已直接写入）；IDM 返回动作
                vehicle.before_step(action)
            if actor.mode == "script" and step >= actor.plan.handover_step:
                self._handover(actor)  # 机动结束 -> 转为普通交通，避免"停车/消失"式悖论
        self._update_windows(step)
        _refresh_actors(self.engine, self)
        return dict()

    def after_step(self, *args, **kwargs) -> dict:
        if not self._actors:
            return dict()
        objects = self.engine.get_objects()
        for actor in list(self._actors.values()):
            vehicle = objects.get(actor.vehicle_id)
            if vehicle is not None:
                vehicle.after_step()
        self._update_windows(int(self.engine.episode_step))
        _refresh_actors(self.engine, self)
        return dict()

    def _handover(self, actor: _Actor) -> None:
        """机动结束后把脚本车交给 IDMPolicy，使其继续作为普通交通行驶。"""
        vehicle = self.engine.get_objects().get(actor.vehicle_id)
        if vehicle is None:
            return
        old = self.engine.get_policy(actor.vehicle_id)
        if old is not None:
            old.destroy()
        self.add_policy(actor.vehicle_id, IDMPolicy, vehicle, self.generate_seed())
        actor.mode = "idm"
        event = self.state["events"].get(actor.key)
        if event is not None:
            event["handover_step"] = int(self.engine.episode_step)

    def destroy(self) -> None:
        self._actors = {}
        self._spec = None
        self._requests = {}
        super().destroy()

    # ---- 生成 / 清理 ----
    def spawn_actors_if_needed(self) -> None:
        if self._spec is None or self._installed_epoch is not None:
            return
        if self.engine.current_map is None:
            return
        self.spawn_actors()

    def spawn_actors(self) -> None:
        """构建计划并生成脚本车；ego 未就绪时静默跳过（下一次 before_step 重试）。"""
        ego = ego_vehicle(self.engine)
        if ego is None:
            return
        self._clear_actors()  # 先清空（含上次部分失败的重试路径），避免重复车/占位误判
        plans = _build_plans(self.engine, self._spec, ego)
        if not plans:
            return
        step = int(self.engine.episode_step)
        self.state["spec_id"] = spec_field(self._spec, "id")
        self.state["seed"] = spec_field(self._spec, "seed")
        self.state["spawn_step"] = step
        degraded_any = False
        for key, plan in plans.items():
            plan.spawn_step = step
            vehicle = self.spawn_object(DefaultVehicle, vehicle_config=plan.vehicle_config())
            lane = plan.base_lane
            position, heading = plan.pose(lane, plan.last_ego_long, 0.0)
            vehicle.set_position(position)
            vehicle.set_heading_theta(float(heading))
            vehicle.set_velocity(plan.velocity(heading))
            vehicle.set_angular_velocity(0.0)
            self.add_policy(vehicle.name, WaypointPolicy, vehicle, plan)
            self._actors[key] = _Actor(key=key, kind=plan.kind, vehicle_id=vehicle.name, plan=plan)
            event = self.state["events"].get(key)
            if event is None:
                event = _event_entry(plan.kind, key)
                self.state["events"][key] = event
            event.update(
                start_step=int(plan.window_start_step),
                end_step=int(plan.window_end_step),
                duration_steps=int(plan.window_end_step - plan.window_start_step),
                phase="pending",
                active=False,
                fired=False,
                degraded=bool(plan.degraded),
                actor_id=vehicle.name,
                actor_alive=True,
                handover_step=int(plan.handover_step),
                side=plan.side,
                gap_m=float(plan.gap_m),
            )
            degraded_any = degraded_any or bool(plan.degraded)
        self.state["degraded"] = degraded_any
        self._installed_epoch = step
        if step == 0:  # reset 立即返回时，让窗口在第 0 步就可读
            self._update_windows(step)
        _refresh_actors(self.engine, self)

    def _clear_actors(self) -> None:
        ids = [actor.vehicle_id for actor in self._actors.values() if actor.vehicle_id in self.spawned_objects]
        self._actors = {}
        if ids:
            self.clear_objects(ids)

    def _update_windows(self, step: int) -> None:
        for key, event in self.state["events"].items():
            start, end = event.get("start_step"), event.get("end_step")
            active = bool(start is not None and end is not None and int(start) <= step < int(end))
            event["active"] = active
            if active and self._actors:
                event["fired"] = True
            actor = self._actors.get(key)
            event["phase"] = self._phase(actor.plan, step) if actor is not None else "pending"

    @staticmethod
    def _phase(plan: ActorPlan, step: int) -> str:
        t = max(0.0, (int(step) - int(plan.spawn_step))) * STEP_DT
        if t < plan.t_trigger:
            return "approach"
        if t < plan.t_trigger + plan.merge_duration:
            return "merge"
        return "hold"


# --------------------------------------------------------------------------------------
# install / reset 钩子
# --------------------------------------------------------------------------------------


def _runtime(env: Any) -> dict:
    runtime = getattr(env, "_scenario_runtime", None)
    if not isinstance(runtime, dict):
        runtime = {"spec": None, "manager": None, "reset_patched": False}
        try:
            setattr(env, "_scenario_runtime", runtime)
        except Exception:  # noqa: BLE001 - 某些 wrapper 可能禁用属性赋值；退化为不缓存
            pass
    return runtime


def _ensure_manager(env: Any) -> Optional[ScenarioBehaviorManager]:
    """幂等地创建/取回管理器（需要引擎已初始化），并把事件状态别名到 env 上。"""
    runtime = _runtime(env)
    engine = getattr(env, "engine", None)
    if engine is None:
        return None
    manager = runtime.get("manager")
    if not isinstance(manager, ScenarioBehaviorManager):
        manager = getattr(engine, MANAGER_NAME, None)
        if not isinstance(manager, ScenarioBehaviorManager):
            manager = ScenarioBehaviorManager()
            engine.register_manager(MANAGER_NAME, manager)
        runtime["manager"] = manager
    try:
        env._scenario_events = manager.state
    except Exception:  # noqa: BLE001
        pass
    engine._scenario_events = manager.state
    engine._scenario_behavior_spec = runtime.get("spec")
    return manager


def _patch_reset(env: Any) -> None:
    """引擎未建时包装 ``env.reset``：首次 reset 前先注册管理器，保证 reset 内后处理能生成脚本车。"""
    runtime = _runtime(env)
    if runtime.get("reset_patched"):
        return
    runtime["reset_patched"] = True
    original = env.reset

    def _patched_reset(*args, **kwargs):
        if getattr(env, "engine", None) is None:
            try:
                env.lazy_init()
            except Exception:  # noqa: BLE001 - 把错误留给原始 reset 暴露
                pass
        manager = _ensure_manager(env)
        if manager is not None and runtime.get("spec") is not None:
            # spawn_now=False：本次 reset 由引擎生命周期负责生成（after_reset / install 钩子）
            manager.install(runtime["spec"], spawn_now=False)
        return original(*args, **kwargs)

    try:
        env.reset = _patched_reset
    except Exception:  # noqa: BLE001
        pass


def install(env: Any, spec: Any) -> None:
    """挂载脚本行为：``build_env``（L2）在每次 ``reset()`` 的后处理阶段调用。

    - 引擎/地图已就绪（L2 的实际调用点）：登记 spec 后立即生成脚本车；
    - 引擎未就绪（直接 ``install`` 后再 ``reset``）：包装 reset，在首次 reset 时注册并生成。
    """
    runtime = _runtime(env)
    runtime["spec"] = spec
    manager = _ensure_manager(env)
    if manager is None:  # 引擎尚未初始化
        _patch_reset(env)
        return
    manager.install(spec, spawn_now=False)
    if env.engine.current_map is not None:
        manager.spawn_actors_if_needed()
