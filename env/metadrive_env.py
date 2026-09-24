"""MetaDrive 环境封装（L2）：显式 block 建图 + 建图后处理重放 + 可选地图 LRU。

职责
----
1. 用 spec 的显式 block 序列建 ``MetaDriveEnv``：``store_map=False``、``preload_models=False``、
   关闭一切渲染/传感器开销（自车观测由 ``env.obs.builder.ObservationBuilder`` 提供，
   所以 ``agent_observation=DummyObservation``，避免每步白算 240 线 lidar）；
2. 建图后逐 lane 写限速（spec.limits，单位 **m/s**）与可选线型覆盖，并调用
   ``env.scenario.behaviors.install(env, spec)``（L1b，惰性导入）；
3. 可选 ≤32 张地图 LRU（默认关）。

**关键陷阱**：``store_map=False`` 时每次 ``env.reset()`` 都会重建地图，后处理与 behaviors
必须在每次重建后重放。因此本模块用 ``MetaDriveEnv`` 子类覆写 ``reset()``，
而不是在 ``build_env`` 里做一次性后处理——否则 validator / 训练循环里直接调 ``env.reset()``
的路径会漏掉限速（P0 报告 §5 明确要求验证"每次重建后的重放"）。

spec 字段的宽容读取（L1a 的 ScenarioSpec 可能尚未就绪，全部走 getattr / 鸭子类型）::

    spec.seed: int
    spec.blocks: "SCXRO"（或单字符列表）
    spec.traffic: {"density": 0.2, ...} 或直接 float
    spec.limits: 见 _parse_limits 的说明（m/s）

限速单位说明：MetaDrive 的 ``lane.speed_limit`` 上游单位混乱——PG block 传的是 m/s
（``component/pgblock/ramp.py:33`` "12 m/s ~= 40 km/h"），而 map features 导出字段名叫
``speed_limit_kmh``（``component/map/pg_map.py:152``）。本项目统一用 **m/s** 写入/读取。
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.road_network.road import Road
from metadrive.constants import DEFAULT_AGENT
from metadrive.envs import MetaDriveEnv
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.obs.observation_base import DummyObservation
from metadrive.type import MetaDriveType

if TYPE_CHECKING:  # 仅类型标注；L1a 未就绪时不影响导入
    from env.scenario.spec import ScenarioSpec

logger = logging.getLogger(__name__)

#: 地图 LRU 上限（P0 报告 §3：≈3.1 MB/张，32 张 ≈ 100 MB/进程）
MAX_LRU_SIZE = 32
DEFAULT_LRU_SIZE = 0
#: AbstractLane 的"未设置"默认限速（component/lane/abs_lane.py:22）
SPEED_LIMIT_UNSET = 1000.0
#: 兜底合法 block 字符（V2 分布可用集：S C r R y Y X T O U B $；见 L1a taxonomy.VALID_BLOCK_CHARS）
_BLOCK_ID_CHARS = frozenset("SCrRyYXT OUB$".replace(" ", ""))
#: taxonomy 不可用时的几何标签 -> block 字符兜底表（与 L1a taxonomy.BLOCK_CHARS 同向）
_FALLBACK_GEOMETRY_BLOCK_CHARS: dict[str, str] = {
    "straight": "S",
    "curve": "C",
    "ramp_in": "r",
    "ramp_out": "R",
    "merge": "y",
    "split": "Y",
    "intersection": "X",
    "t_intersection": "T",
    "roundabout": "O",
    "uturn": "U",
    "bidirection": "B",
    "tollgate": "$",
}
#: 默认自车 spawn 车道（与 METADRIVE_DEFAULT_CONFIG.agent_configs 一致）
_DEFAULT_AGENT_SPAWN_LANE = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)


# ======================================================================================
# spec 宽容读取
# ======================================================================================
def _spec_seed(spec) -> int:
    seed = getattr(spec, "seed", None)
    if seed is None:
        raise ValueError("spec.seed 缺失：无法确定 MetaDrive scenario seed")
    return int(seed)


def _valid_block_chars() -> frozenset[str]:
    """合法 block 字符；优先用 L1a ``taxonomy.VALID_BLOCK_CHARS``（惰性导入）。"""
    try:
        from env.scenario.taxonomy import VALID_BLOCK_CHARS

        chars = frozenset(str(c) for c in VALID_BLOCK_CHARS if isinstance(c, str) and len(c) == 1)
        if chars:
            return chars
    except Exception:  # noqa: BLE001 - taxonomy 未就绪时用兜底集
        pass
    return _BLOCK_ID_CHARS


def _spec_blocks(spec) -> str:
    blocks = getattr(spec, "blocks", None)
    if isinstance(blocks, str) and blocks:
        sequence = blocks
    elif isinstance(blocks, (list, tuple)) and blocks:
        chars = []
        for item in blocks:
            if isinstance(item, str) and len(item) == 1:
                chars.append(item)
            elif hasattr(item, "ID"):
                chars.append(str(item.ID))
            else:
                raise ValueError(f"spec.blocks 元素无法识别：{item!r}")
        sequence = "".join(chars)
    else:
        raise ValueError("spec.blocks 必须是非空显式 block 序列（如 'SCXRO'）")
    if sequence.startswith("I"):
        # BIG 会自己前置 FirstPGBlock（ID="I"，component/algorithm/BIG.py:86）；容错去掉
        sequence = sequence[1:]
    valid = _valid_block_chars()
    bad = sorted(set(sequence) - valid)
    if bad:
        raise ValueError(f"spec.blocks={sequence!r} 含非法 block 字符 {bad}；合法集合={sorted(valid)}")
    if not sequence:
        raise ValueError("spec.blocks 去掉首块 I 后为空")
    return sequence


def _spec_traffic_density(spec) -> float:
    traffic = getattr(spec, "traffic", None)
    if isinstance(traffic, (int, float)) and not isinstance(traffic, bool):
        return float(traffic)
    if isinstance(traffic, dict):
        for key in ("density", "traffic_density"):
            if key in traffic:
                try:
                    return float(traffic[key])
                except (TypeError, ValueError):
                    continue
    return 0.0


def _spec_random_traffic(spec) -> bool:
    """spec.traffic["random_traffic"] -> MetaDrive 同名配置（缺省 False，保持可复现）。"""
    traffic = getattr(spec, "traffic", None)
    if isinstance(traffic, dict) and isinstance(traffic.get("random_traffic"), bool):
        return bool(traffic["random_traffic"])
    return False


def _spec_ego_overrides(spec) -> dict:
    """把 ``spec.ego`` 的 spawn 参数映射成 MetaDrive ``agent_configs`` 覆盖项。

    为什么 ``spawn_velocity`` 要转换：spec 里是标量 m/s，而 MetaDrive 的 ``spawn_velocity``
    实际是**速度向量**——``BaseVehicle.reset`` 直接把它交给 ``set_velocity`` 并取 ``direction[0]``
    （base_vehicle.py:397-398），传标量会 ``TypeError``；上游用法见
    ``manager/scenario_map_manager.py:95``（传的是 ``init_state["velocity"]`` 向量）。
    这里转成车体系 ``[v, 0]`` 且 ``spawn_velocity_car_frame=True``：reset 时先设 heading
    （base_vehicle.py:355，取自 spawn 车道航向）再设速度，因此该向量就是沿车道前进的速度。
    """
    ego = getattr(spec, "ego", None)
    if not isinstance(ego, dict):
        return {}
    overrides: dict = {}
    lane_index = ego.get("spawn_lane_index")
    if lane_index is not None:
        overrides["spawn_lane_index"] = tuple(lane_index)
    longitude = ego.get("spawn_longitude")
    if isinstance(longitude, (int, float)) and not isinstance(longitude, bool):
        overrides["spawn_longitude"] = float(longitude)
    lateral = ego.get("spawn_lateral")
    if isinstance(lateral, (int, float)) and not isinstance(lateral, bool):
        overrides["spawn_lateral"] = float(lateral)
    velocity = ego.get("spawn_velocity")
    if isinstance(velocity, (int, float)) and not isinstance(velocity, bool):
        if float(velocity) > 0.0:
            overrides["spawn_velocity"] = [float(velocity), 0.0]
            overrides["spawn_velocity_car_frame"] = True
    elif isinstance(velocity, (list, tuple)) and len(velocity) == 2:
        overrides["spawn_velocity"] = [float(velocity[0]), float(velocity[1])]
    return overrides


def _geometry_to_block_char() -> dict[str, str]:
    """几何标签 -> block 字符；优先用 L1a 的 taxonomy.BLOCK_CHARS（惰性导入）。"""
    mapping = dict(_FALLBACK_GEOMETRY_BLOCK_CHARS)
    try:
        from env.scenario.taxonomy import BLOCK_CHARS

        for label, char in dict(BLOCK_CHARS).items():
            if isinstance(char, str) and len(char) == 1:
                mapping[str(label).lower()] = char
    except Exception:  # noqa: BLE001 - taxonomy 未就绪时用兜底表
        pass
    return mapping


# ======================================================================================
# 限速 / 线型后处理
# ======================================================================================
@dataclass
class _LimitRules:
    """从 spec.limits 解析出的规则（优先级：by_lane > by_block > default）。"""

    default: float | None = None
    by_block: dict[str, float] = field(default_factory=dict)
    by_lane: dict[tuple, float] = field(default_factory=dict)
    line_types: dict = field(default_factory=dict)
    unrecognized: list = field(default_factory=list)


def _as_float(value) -> float | None:
    """把标量/``{"speed_limit": x}`` 解析成正数 m/s；其余返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    if isinstance(value, dict):
        for key in ("speed_limit", "limit", "value", "default"):
            if key in value:
                return _as_float(value[key])
    return None


def _is_block_char(text: str) -> bool:
    """单字符 block 标识（S/C/X/…/$/）；不锁定具体集合，避免上游新增 block 后静默失效。"""
    return len(text) == 1 and (text.isalnum() or text == "$")


def _add_block_entry(rules: _LimitRules, key: str, value, labels: dict[str, str]) -> None:
    limit = _as_float(value)
    if limit is None:
        return
    text = str(key).strip()
    if _is_block_char(text):
        rules.by_block[text] = limit
    elif text.lower() in labels:
        rules.by_block[labels[text.lower()]] = limit
    else:
        rules.unrecognized.append(text)


def _add_lane_entry(rules: _LimitRules, key, value) -> None:
    limit = _as_float(value)
    if limit is None:
        return
    index = key
    if isinstance(index, str):
        try:  # JSON 往返会把元组键变成字符串 "(from, to, id)"
            parsed = ast.literal_eval(index.strip())
        except Exception:  # noqa: BLE001
            parsed = None
        index = parsed
    if isinstance(index, (list, tuple)) and len(index) == 3:
        rules.by_lane[tuple(index)] = limit
    else:
        rules.unrecognized.append(str(key))


def _parse_limits(limits) -> _LimitRules:
    """解析 spec.limits（单位 m/s）。

    支持（可混用）::

        limits = 8.3                                    # 标量：所有 lane
        limits = {"default": 8.3}                       # 或 "speed_limit"/"lane_speed_limit"
        limits = {"speed_limit_mps": 8.3}               # L1a 规范：场景级最保守限速（默认值）
        limits = {"by_char": {"S": 13.9, "C": 8.0}}     # L1a 规范：按 block 字符
        limits = {"S": 13.9, "C": 8.0, "r": 11.0}       # 等价简写
        limits = {"straight": 13.9, "curve": 8.0}       # 按几何标签（taxonomy.BLOCK_CHARS 反查）
        limits = {"by_block": {...}}                    # 同上，显式分组
        limits = {"by_lane": {("0S0-0-", "0S0-1-", 0): 8.3}}
        limits = {"line_types": {"C": ["ROAD_LINE_SOLID_SINGLE_YELLOW", "ROAD_EDGE_BOUNDARY"]}}

    未识别的键会记录在 ``unrecognized`` 里并由 :meth:`SpecMetaDriveEnv.apply_postprocess` 告警。
    """
    rules = _LimitRules()
    if limits is None:
        return rules
    if isinstance(limits, (int, float)) and not isinstance(limits, bool):
        rules.default = _as_float(limits)
        return rules
    if not isinstance(limits, dict):
        rules.unrecognized.append(type(limits).__name__)
        return rules

    labels = _geometry_to_block_char()
    for key, value in limits.items():
        if isinstance(key, tuple) and len(key) == 3:
            _add_lane_entry(rules, key, value)
            continue
        if not isinstance(key, str):
            rules.unrecognized.append(repr(key))
            continue
        low = key.strip().lower()
        if low in ("line_types", "lines", "line_type"):
            rules.line_types = dict(value) if isinstance(value, dict) else {}
        elif low in ("default", "speed_limit", "lane_speed_limit", "default_speed_limit", "limit",
                     "speed_limit_mps"):
            limit = _as_float(value)
            if limit is not None:
                rules.default = limit
        elif low in ("by_block", "blocks", "block", "by_char", "chars"):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _add_block_entry(rules, str(sub_key), sub_value, labels)
        elif low in ("by_lane", "lanes", "lane"):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _add_lane_entry(rules, sub_key, sub_value)
        elif low in labels:
            limit = _as_float(value)
            if limit is not None:
                rules.by_block[labels[low]] = limit
        elif _is_block_char(key):
            limit = _as_float(value)
            if limit is not None:
                rules.by_block[key] = limit
        elif key.strip().startswith("("):
            _add_lane_entry(rules, key, value)
        else:
            rules.unrecognized.append(key)
    return rules


def _lane_block_char(lane) -> str | None:
    """车道所属 block 字符（Road.block_ID()，road.py:42-47；负向道路也能取到）。

    不做"合法集合"过滤：只要 road network 给出单字符就返回，交由规则匹配决定是否命中；
    这样上游新增 block 类型时 L2 不会静默漏掉限速。
    """
    index = getattr(lane, "index", None)
    if index is None or len(index) < 2:
        return None
    try:
        char = Road(str(index[0]), str(index[1])).block_ID()
    except Exception:  # noqa: BLE001
        return None
    return char if isinstance(char, str) and len(char) == 1 else None


def _lane_speed_limit(lane, rules: _LimitRules) -> float | None:
    index = getattr(lane, "index", None)
    if index is not None:
        key = tuple(index)
        if key in rules.by_lane:
            return rules.by_lane[key]
    char = _lane_block_char(lane)
    if char is not None and char in rules.by_block:
        return rules.by_block[char]
    return rules.default


def _to_line_type(value) -> str | None:
    """把 int id / MetaDrive 类型字符串 / PGLineType 名转换成线型字符串。"""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        from env.obs.ld import LINE_TYPE_IDS

        return {v: k for k, v in LINE_TYPE_IDS.items()}.get(int(value))
    text = str(value)
    from env.obs.ld import LINE_TYPE_IDS

    if text in LINE_TYPE_IDS:
        return text
    alias = {
        "none": MetaDriveType.LINE_UNKNOWN,
        "broken": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
        "continuous": MetaDriveType.LINE_SOLID_SINGLE_WHITE,
        "solid": MetaDriveType.LINE_SOLID_SINGLE_WHITE,
        "side": MetaDriveType.BOUNDARY_LINE,
        "guardrail": MetaDriveType.GUARDRAIL,
    }
    return alias.get(text.strip().lower())


def _line_type_pair(value) -> tuple[str, str] | None:
    """(left, right)；标量表示两侧相同。"""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        left, right = _to_line_type(value[0]), _to_line_type(value[1])
        return (left, right) if left is not None and right is not None else None
    single = _to_line_type(value)
    return (single, single) if single is not None else None


def _apply_line_type_overrides(lanes: Sequence, rules: _LimitRules) -> int:
    """按 block 字符 / 几何标签 / lane 索引覆盖左右线型；返回生效的 lane 数。"""
    if not rules.line_types:
        return 0
    labels = _geometry_to_block_char()
    label_by_char = {char: label for label, char in labels.items()}
    count = 0
    for lane in lanes:
        override = None
        index = getattr(lane, "index", None)
        if index is not None and tuple(index) in rules.line_types:
            override = rules.line_types[tuple(index)]
        else:
            char = _lane_block_char(lane)
            if char is not None:
                if char in rules.line_types:
                    override = rules.line_types[char]
                elif label_by_char.get(char) in rules.line_types:
                    override = rules.line_types[label_by_char[char]]
        pair = _line_type_pair(override)
        if pair is None:
            continue
        lane.line_types = [pair[0], pair[1]]  # [0]=左, [1]=右（pg_block.py:257-258）
        count += 1
    return count


# ======================================================================================
# 可选地图 LRU（默认关）
# ======================================================================================
class _LruMapManager(PGMapManager):
    """``store_map=False`` + 有界地图 LRU：命中则直接复用已建好的地图对象。

    为什么缓存"活地图"而不是序列化 meta data：MetaDrive 自带的 ``store_map=True`` 就是
    复用活对象（pg_map_manager.py:56-66），无需走 ``PG_MAP_FILE`` 重建路径，零序列化风险；
    代价是 ≈3.1 MB/张（P0 报告 §3），所以上限 32 且默认关闭。
    """

    def __init__(self, max_entries: int = MAX_LRU_SIZE):
        super().__init__()
        self.max_entries = max(1, int(max_entries))
        self._lru: dict[int, PGMap] = {}

    def reset(self) -> None:
        seed = int(self.engine.global_seed)
        cached = self._lru.pop(seed, None)
        if cached is not None:
            self._lru[seed] = cached  # 命中：放回队尾（最近使用）
            self.load_map(cached)
            return
        super().reset()  # store_map=False：现建现用，self.maps 不保留
        if self.current_map is not None:
            self._lru[seed] = self.current_map
            self._evict()

    def before_reset(self) -> None:
        # 不能调用 PGMapManager.before_reset：store_map=False 时它会 destroy 地图，
        # 而地图可能仍在本 LRU 中待复用。
        if self.current_map is not None:
            self.current_map.detach_from_world()
            self.current_map = None

    def _evict(self) -> None:
        # dict 保序（Python 3.7+）；跳过当前地图，最多遍历一遍防止病态死循环
        guard = len(self._lru)
        while len(self._lru) > self.max_entries and guard > 0:
            seed, map_ = next(iter(self._lru.items()))
            if map_ is self.current_map:
                self._lru.pop(seed)
                self._lru[seed] = map_
            else:
                self._lru.pop(seed)
                map_.detach_from_world()
                map_.destroy()
            guard -= 1

    def destroy(self) -> None:
        for map_ in list(self._lru.values()):
            map_.detach_from_world()
            map_.destroy()
        self._lru.clear()
        super().destroy()


# ======================================================================================
# env 子类：保证任何 reset 都重放建图后处理
# ======================================================================================
class SpecMetaDriveEnv(MetaDriveEnv):
    """MetaDriveEnv 子类：绑定 spec，并在**每次** reset（=每次重建地图）后重放后处理。"""

    def __init__(self, config: dict | None = None, *, map_lru_size: int = DEFAULT_LRU_SIZE):
        self._spec: Any = None
        requested = int(map_lru_size or 0)
        if requested > MAX_LRU_SIZE:
            logger.warning("map_lru_size=%d 超过上限 %d，已截断", requested, MAX_LRU_SIZE)
        self._map_lru_size = max(0, min(requested, MAX_LRU_SIZE))
        self.behavior_install_error: str | None = None
        self.postprocess_stats: dict | None = None
        super().__init__(config)

    # ------------------------------------------------------------------ 绑定
    def bind_spec(self, spec) -> "SpecMetaDriveEnv":
        """绑定/更新 spec（同一 env 复用不同 spec 时也会生效）。"""
        self._spec = spec
        return self

    @property
    def spec(self):
        return self._spec

    @property
    def map_lru_size(self) -> int:
        """生效的地图 LRU 容量（0=关闭）。"""
        return self._map_lru_size

    # ------------------------------------------------------------------ 生命周期
    def setup_engine(self) -> None:
        super().setup_engine()
        if self._map_lru_size > 0:
            self.engine.update_manager("map_manager", _LruMapManager(self._map_lru_size))

    def reset(self, seed=None):
        result = super().reset(seed)
        # 关键：store_map=False 时上面这次 reset 已经重建了地图，后处理必须紧跟其后重放
        self.apply_postprocess()
        self.install_behaviors()
        return result

    # ------------------------------------------------------------------ 后处理
    def apply_postprocess(self) -> dict:
        """逐 lane 写限速（+ 可选线型覆盖）。可重复调用（幂等）；每次 reset 后由钩子调用。"""
        stats = {"lanes": 0, "speed_limit_set": 0, "unmatched": 0, "line_type_set": 0}
        self.postprocess_stats = stats
        spec = self._spec
        if spec is None:
            return stats
        map_ = self.current_map
        if map_ is None:
            return stats

        lanes = list(map_.road_network.get_all_lanes())
        stats["lanes"] = len(lanes)
        rules = _parse_limits(getattr(spec, "limits", None))
        for lane in lanes:
            limit = _lane_speed_limit(lane, rules)
            if limit is None:
                if float(getattr(lane, "speed_limit", 0.0)) >= SPEED_LIMIT_UNSET:
                    stats["unmatched"] += 1
                continue
            lane.set_speed_limit(float(limit))
            stats["speed_limit_set"] += 1
        if rules.line_types:
            stats["line_type_set"] = _apply_line_type_overrides(lanes, rules)
        if stats["unmatched"] or rules.unrecognized:
            logger.warning(
                "spec=%s: 未匹配限速的 lane=%d，未识别 limits 键=%s（保留 %s=未设置）",
                getattr(spec, "id", "?"),
                stats["unmatched"],
                rules.unrecognized,
                SPEED_LIMIT_UNSET,
            )
        return stats

    def install_behaviors(self) -> None:
        """安装脚本化 cut-in/cut-out（L1b）。

        惰性/防御式导入：L1b 未就绪（模块缺失）时跳过；``install`` 抛异常时记录到
        ``self.behavior_install_error`` 并告警——观测/建图不应因为行为脚本的 bug 而整体不可用，
        validator 可通过该属性看到失败原因。
        """
        spec = self._spec
        if spec is None:
            return
        try:
            from env.scenario import behaviors
        except ImportError:
            self.behavior_install_error = "env.scenario.behaviors 不可用（L1b 未就绪）"
            logger.info("跳过 behaviors.install：%s", self.behavior_install_error)
            return
        install = getattr(behaviors, "install", None)
        if install is None:
            self.behavior_install_error = "env.scenario.behaviors.install 不存在"
            logger.warning("%s", self.behavior_install_error)
            return
        try:
            install(self, spec)
            self.behavior_install_error = None
        except Exception as exc:  # noqa: BLE001
            self.behavior_install_error = f"{type(exc).__name__}: {exc}"
            logger.warning("behaviors.install 失败（已记录到 env.behavior_install_error）：%s", exc)


# ======================================================================================
# 公开接口
# ======================================================================================
def build_env(
    spec,
    *,
    traffic_density: float | None = None,
    use_render: bool = False,
    lru_size: int = DEFAULT_LRU_SIZE,
    seed_pool: tuple[int, int] | None = None,
) -> SpecMetaDriveEnv:
    """按 spec 建环境（不 reset）。

    - ``traffic_density=None`` 时取 ``spec.traffic["density"]``（缺失按 0.0）；
    - ``lru_size`` 打开 ≤32 张地图 LRU（默认 0=关，对应 P0 "不做地图缓存"的决定）；
    - ``seed_pool=(start, stop)``：把 env 配成多场景（``num_scenarios=stop-start``），
      同一个 env 可在池内反复 ``reset(seed)`` —— 这是让 LRU 有意义的用法（P0 §5：P2 小课程集
      反复采样）。要求池内所有 spec 的 ``blocks`` 相同（地图配置在构造期固化），
      且每次 reset 前用 :meth:`SpecMetaDriveEnv.bind_spec` 更新 spec（MetaDriveWrapper 已处理）；
    - 限速/线型后处理与 behaviors.install 在每次 ``env.reset()`` 后自动重放。
    """
    blocks = _spec_blocks(spec)
    seed = _spec_seed(spec)
    if seed_pool is None:
        num_scenarios, start_seed = 1, seed
    else:
        pool_start, pool_stop = int(seed_pool[0]), int(seed_pool[1])
        if not (pool_start <= seed < pool_stop):
            raise ValueError(f"spec.seed={seed} 不在 seed_pool=({pool_start}, {pool_stop}) 内")
        num_scenarios, start_seed = pool_stop - pool_start, pool_start
    density = _spec_traffic_density(spec) if traffic_density is None else float(traffic_density)

    # 自车 spawn：spec.ego 与 MetaDrive agent_configs 同名键；未给车道时保持默认随机车道
    ego_overrides = _spec_ego_overrides(spec)
    agent_config = dict(use_special_color=True, spawn_lane_index=_DEFAULT_AGENT_SPAWN_LANE)
    agent_config.update(ego_overrides)

    config = dict(
        use_render=bool(use_render),
        log_level=50,  # CRITICAL：与 P0 recon 冻结口径一致（tools/measure/api_recon.py）
        num_scenarios=num_scenarios,
        start_seed=start_seed,
        map=blocks,  # str -> BIG_BLOCK_SEQUENCE（component/map/pg_map.py:28-37）
        traffic_density=density,
        random_traffic=_spec_random_traffic(spec),
        # spec 指定了 spawn 车道就不要再随机（agent_manager.random_spawn_lane_in_single_agent 会覆盖）
        random_spawn_lane_index="spawn_lane_index" not in ego_overrides,
        agent_configs={DEFAULT_AGENT: agent_config},
        preload_models=False,
        store_map=False,
        # 自车观测走 ObservationBuilder；DummyObservation 省掉每步 lidar/state 计算
        agent_observation=DummyObservation,
        vehicle_config=dict(
            lidar=dict(num_lasers=0),
            side_detector=dict(num_lasers=0),
            lane_line_detector=dict(num_lasers=0),
            show_navi_mark=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            show_line_to_navi_mark=False,
            show_navigation_arrow=False,
        ),
    )
    env = SpecMetaDriveEnv(config, map_lru_size=lru_size)
    env.bind_spec(spec)
    logger.info(
        "build_env: seed=%s blocks=%s density=%.3f lru=%s scenarios=[%s,%s)",
        seed,
        blocks,
        density,
        env.map_lru_size,
        start_seed,
        start_seed + num_scenarios,
    )
    return env


class MetaDriveWrapper:
    """可选薄封装：同一进程内按 spec 复用 env，spec 的 (blocks, seed, spawn 参数) 变化时重建。

    ``seed_pool=(start, stop)`` 时用多场景 env 承载池内所有 spec（blocks 必须一致），
    配合 ``lru_size`` 才能真正吃到地图缓存。
    """

    def __init__(
        self,
        *,
        traffic_density: float | None = None,
        use_render: bool = False,
        lru_size: int = DEFAULT_LRU_SIZE,
        seed_pool: tuple[int, int] | None = None,
    ):
        self.traffic_density = traffic_density
        self.use_render = bool(use_render)
        self.lru_size = int(lru_size or 0)
        self.seed_pool = None if seed_pool is None else (int(seed_pool[0]), int(seed_pool[1]))
        self._env: SpecMetaDriveEnv | None = None
        self._key: tuple | None = None
        self.spec = None

    @property
    def env(self) -> SpecMetaDriveEnv | None:
        return self._env

    def _key_for(self, spec) -> tuple:
        """复用键：blocks/spawn 参数/密度任一变化都必须重建（agent_config 是构建期固化的）。

        配置了 seed_pool 时 seed 不进 key（池内共享同一 env），否则 seed 参与。
        """
        density = _spec_traffic_density(spec) if self.traffic_density is None else float(self.traffic_density)
        seed_key = self.seed_pool if self.seed_pool is not None else _spec_seed(spec)
        return (_spec_blocks(spec), seed_key, repr(_spec_ego_overrides(spec)), density)

    def reset(self, spec):
        """建/复用 env 并 reset；返回 MetaDrive 的 ``(obs, info)``（obs 为占位）。"""
        key = self._key_for(spec)
        if self._env is None or key != self._key:
            self.close()
            self._env = build_env(
                spec,
                traffic_density=self.traffic_density,
                use_render=self.use_render,
                lru_size=self.lru_size,
                seed_pool=self.seed_pool,
            )
            self._key = key
        self.spec = spec
        self._env.bind_spec(spec)  # 同 seed/blocks 但 limits/behaviors 变了也能生效
        return self._env.reset(seed=_spec_seed(spec))

    def step(self, action):
        if self._env is None:
            raise RuntimeError("MetaDriveWrapper.step 前必须先 reset(spec)")
        return self._env.step(action)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._key = None

    def __enter__(self) -> "MetaDriveWrapper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
