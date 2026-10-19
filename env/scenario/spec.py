"""场景 spec 数据契约：字段、JSON 往返与自检。

用途
----
generator 产出、validator / env 封装消费；落盘为 ``env/specs/*.json``（已 gitignore）。
只依赖标准库与本包 taxonomy，import 无副作用。

关键接口
--------
- ``ScenarioSpec``: 字段见 dataclass；``to_dict`` / ``from_dict`` / ``validate``
- ``load_specs(path)`` / ``save_specs(specs, path)``: JSON 读写
- 落盘格式: ``{"schema_version": 1, "count": N, "specs": [...]}``；``load_specs`` 也接受裸列表

字段单位约定（跨线必须一致）
--------------------------
- ``limits``：m/s，L2 建图后逐 lane 写入 ``lane.set_speed_limit``。MetaDrive 自带 block 也用
  m/s（ramp.py:33 "12 m/s"、tollgate.py:19 "3 m/s"），但其 ``BaseVehicle.overspeed``
  （base_vehicle.py:951）拿 ``speed_km_h`` 比较，属上游不一致，不要据此改单位。
- ``traffic["events"][*]["speed_mps"]``、``ego["spawn_velocity"]``：m/s
  （MetaDrive ``spawn_velocity`` 即 m/s，base_env.py:147）。
- ``traffic["events"][*]["trigger_step"] / duration_steps``：env step（0.1 s/step；
  physics_world_step_size=2e-2 × decision_repeat=5，base_env.py:188-190）。
- ``ego["spawn_longitude"]``：沿 spawn 车道的纵向偏移（m），须 < FirstPGBlock.ENTRANCE_LENGTH=10
  （first_block.py:26；默认 spawn_longitude=5.0，base_env.py:142），否则会落到首块主体上。
- ``ego["spawn_lane_index"]``：``(road_start, road_end, lane_id)``。road 键恒为 MetaDrive First
  block 的固定节点（BIG 自动前置，见 ``taxonomy.EGO_SPAWN_ROAD``）；lane_id 0=最左、从左到右
  递增，取值域为 ``taxonomy.EGO_SPAWN_LANE_NUM``（MetaDrive map_config 默认 3 车道）。
  generator 取中间车道，保证 cut_in/cut_out 请求侧的邻车道存在；L2 见该字段即关闭随机
  spawn 车道（metadrive_env.py ``random_spawn_lane_index=False``），spawn 因此完全确定。

内容约束（生成集，validate 对账）
-------------------------------
- ``geometry`` / ``blocks`` 不含 ``bidirection``（``B``）：S→B 接缝几何不可通行（S 块 lane-0
  左黄线与 B 块正向车道右黄线 x 向仅隔 ~2 m，自车底盘 4.5 m，任何路径都跨线 ->
  ``on_yellow_continuous_line`` -> ``out_of_road``），已排除出生成集，见
  ``taxonomy.EXCLUDED_GEOMETRY_LABELS``。
- 含 ``roundabout`` 的几何：``traffic["density"]`` 必须 <= ``taxonomy.ROUNDABOUT_MAX_DENSITY``
  （环岛车流追尾不可规避），上限记录在 ``traffic["density_cap"]``。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .taxonomy import (
    CONTROL_LABELS,
    DIFFICULTIES,
    EGO_SPAWN_LANE_NUM,
    EGO_SPAWN_ROAD,
    GEOMETRY_LABELS,
    NAV_MANEUVER_LABELS,
    ROUNDABOUT_MAX_DENSITY,
    SEQUENCE_MAX_BLOCKS,
    SEQUENCE_MIN_BLOCKS,
    SPLITS,
    TRAFFIC_PATTERNS,
    TURN_CANDIDATES,
    VALID_BLOCK_CHARS,
    junction_indices,
    sequence_for,
)

SCHEMA_VERSION = 1

# 字段顺序即 JSON 键顺序，也用于 from_dict 的完整性检查。
FIELDS = (
    "id",
    "seed",
    "split",
    "blocks",
    "geometry",
    "traffic",
    "limits",
    "nav",
    "ego",
    "difficulty",
    "labels",
)

# 限速上限（m/s）：55 m/s ≈ 198 km/h，超出视为字段写错。
MAX_SPEED_LIMIT_MPS = 55.0
# 事件/初始速度上限（m/s）。
MAX_EVENT_SPEED_MPS = 60.0


@dataclass
class ScenarioSpec:
    """单个场景的完整描述（可 JSON 化、可复现）。

    - ``id``/``seed``/``split``：split 内唯一 id、MetaDrive ``start_seed``、所属集合
    - ``blocks``：显式 BIG_BLOCK_SEQUENCE 字符串（不含自动前置的 First block "I"）
    - ``geometry``：几何标签序列，``sequence_for(geometry) == blocks``
    - ``traffic``：密度 / 交通形态 / 脚本化事件窗口
    - ``limits``：建图后写入的限速（m/s）
    - ``nav``：名义转向序列与其候选（真正可观测转向以运行期 navigation_command 为准）
    - ``ego``：自车 spawn 参数（MetaDrive vehicle_config 同名键）；``spawn_lane_index`` 显式给出
      首块车道（见模块 docstring），使脚本事件的请求侧存在邻车道
    - ``difficulty``：由密度/间隙/交叉口数派生，与 ``labels["difficulty"]`` 一致
    - ``labels``：静态场景标签（router 监督用）
    """

    id: int
    seed: int
    split: str
    blocks: str
    geometry: list[str]
    traffic: dict
    limits: dict
    nav: dict
    ego: dict
    difficulty: str
    labels: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """深拷贝为普通 dict（dataclasses.asdict 对 list/dict 值做 deepcopy）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        """从 dict 构造；缺字段抛 ValueError，未知字段忽略（向前兼容）。"""
        if not isinstance(d, dict):
            raise ValueError(f"ScenarioSpec.from_dict 需要 dict，实际 {type(d).__name__}")
        missing = [key for key in FIELDS if key not in d]
        if missing:
            raise ValueError(f"ScenarioSpec 缺少字段: {missing}")
        return cls(**{key: d[key] for key in FIELDS})

    def validate(self) -> None:
        """字段/取值自检；任何不一致抛 ValueError（消息带 split#id 前缀便于定位）。"""

        def _fail(msg: str) -> None:
            raise ValueError(f"spec[{self.split}#{self.id}] {msg}")

        # ---- 基础字段 ----
        if not isinstance(self.id, int) or isinstance(self.id, bool) or self.id < 0:
            _fail(f"id 必须是非负整数，实际 {self.id!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            _fail(f"seed 必须是非负整数，实际 {self.seed!r}")
        if self.split not in SPLITS:
            _fail(f"split 必须是 {SPLITS} 之一，实际 {self.split!r}")

        # ---- blocks / geometry ----
        if not isinstance(self.blocks, str) or not self.blocks:
            _fail(f"blocks 必须是非空字符串，实际 {self.blocks!r}")
        bad = sorted({char for char in self.blocks if char not in VALID_BLOCK_CHARS})
        if bad:
            _fail(f"blocks 含非法字符 {bad}；合法字符 {VALID_BLOCK_CHARS}")
        if not (SEQUENCE_MIN_BLOCKS <= len(self.blocks) <= SEQUENCE_MAX_BLOCKS):
            _fail(
                f"blocks 长度 {len(self.blocks)} 超出 [{SEQUENCE_MIN_BLOCKS}, {SEQUENCE_MAX_BLOCKS}]"
                "（不含隐式 First block）"
            )
        if not isinstance(self.geometry, list) or not all(isinstance(g, str) for g in self.geometry):
            _fail(f"geometry 必须是字符串列表，实际 {self.geometry!r}")
        unknown = [g for g in self.geometry if g not in GEOMETRY_LABELS]
        if unknown:
            _fail(f"geometry 含未知标签 {unknown}；合法标签 {GEOMETRY_LABELS}")
        try:
            expected = sequence_for(self.geometry)
        except ValueError as exc:
            _fail(f"geometry 非法: {exc}")
            return  # 仅为类型收窄，不会到达
        if expected != self.blocks:
            _fail(f"blocks={self.blocks!r} 与 sequence_for(geometry)={expected!r} 不一致")

        # ---- difficulty ----
        if self.difficulty not in DIFFICULTIES:
            _fail(f"difficulty 必须是 {DIFFICULTIES} 之一，实际 {self.difficulty!r}")

        # ---- limits ----
        if not isinstance(self.limits, dict):
            _fail("limits 必须是 dict")
        by_char = self.limits.get("by_char")
        speed = self.limits.get("speed_limit_mps")
        if not isinstance(by_char, dict) or set(by_char) != set(self.blocks):
            _fail(f"limits['by_char'] 的键必须恰好等于 blocks 字符集 {sorted(set(self.blocks))}，实际 {by_char!r}")
        for char, value in by_char.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= MAX_SPEED_LIMIT_MPS:
                _fail(f"limits['by_char'][{char!r}] 必须是 (0, {MAX_SPEED_LIMIT_MPS}] 内的数值（m/s），实际 {value!r}")
        if isinstance(speed, bool) or not isinstance(speed, (int, float)):
            _fail(f"limits['speed_limit_mps'] 必须是数值，实际 {speed!r}")
        if abs(float(speed) - min(float(v) for v in by_char.values())) > 1e-9:
            _fail(
                f"limits['speed_limit_mps']={speed!r} 必须等于 by_char 最小值 "
                f"{min(by_char.values())!r}（全程采用最保守限速）"
            )

        # ---- traffic ----
        traffic = self.traffic
        if not isinstance(traffic, dict):
            _fail("traffic 必须是 dict")
        density = traffic.get("density")
        if isinstance(density, bool) or not isinstance(density, (int, float)) or not 0 < float(density) <= 1.0:
            _fail(f"traffic['density'] 必须在 (0, 1] 内，实际 {density!r}")
        # 环岛密度上限：含 roundabout 的几何必须钳位并记录 density_cap（见模块 docstring）。
        roundabout = "roundabout" in self.geometry
        density_cap = traffic.get("density_cap")
        if roundabout and density_cap is None:
            _fail(f"含 roundabout 的几何必须记录 traffic['density_cap']（<= {ROUNDABOUT_MAX_DENSITY}）")
        if density_cap is not None:
            if isinstance(density_cap, bool) or not isinstance(density_cap, (int, float)) or not 0 < float(density_cap) <= 1.0:
                _fail(f"traffic['density_cap'] 必须在 (0, 1] 内，实际 {density_cap!r}")
            if float(density) > float(density_cap) + 1e-9:
                _fail(f"traffic['density']={density!r} 超过声明的 traffic['density_cap']={density_cap!r}")
        if roundabout and float(density) > ROUNDABOUT_MAX_DENSITY + 1e-9:
            _fail(
                f"含 roundabout 的 density 必须 <= ROUNDABOUT_MAX_DENSITY={ROUNDABOUT_MAX_DENSITY}"
                f"（环岛车流追尾不可规避），实际 {density!r}"
            )
        patterns = traffic.get("patterns")
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            _fail(f"traffic['patterns'] 必须是字符串列表，实际 {patterns!r}")
        unknown_patterns = [p for p in patterns if p not in TRAFFIC_PATTERNS]
        if unknown_patterns:
            _fail(f"traffic['patterns'] 含未知形态 {unknown_patterns}；合法取值 {TRAFFIC_PATTERNS}")
        if not isinstance(traffic.get("seed"), int) or isinstance(traffic.get("seed"), bool):
            _fail(f"traffic['seed'] 必须是整数，实际 {traffic.get('seed')!r}")
        if not isinstance(traffic.get("random_traffic"), bool):
            _fail(f"traffic['random_traffic'] 必须是 bool，实际 {traffic.get('random_traffic')!r}")
        events = traffic.get("events")
        if not isinstance(events, list):
            _fail(f"traffic['events'] 必须是列表，实际 {events!r}")
        event_types = set()
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                _fail(f"traffic['events'][{index}] 必须是 dict")
            if event.get("type") not in ("cut_in", "cut_out"):
                _fail(f"traffic['events'][{index}]['type'] 必须是 cut_in/cut_out，实际 {event.get('type')!r}")
            event_types.add(event["type"])
            if event.get("side") not in ("left", "right"):
                _fail(f"traffic['events'][{index}]['side'] 必须是 left/right，实际 {event.get('side')!r}")
            if not isinstance(event.get("trigger_step"), int) or isinstance(event.get("trigger_step"), bool) or event["trigger_step"] < 0:
                _fail(f"traffic['events'][{index}]['trigger_step'] 必须是非负整数，实际 {event.get('trigger_step')!r}")
            if not isinstance(event.get("duration_steps"), int) or isinstance(event.get("duration_steps"), bool) or event["duration_steps"] <= 0:
                _fail(f"traffic['events'][{index}]['duration_steps'] 必须是正整数，实际 {event.get('duration_steps')!r}")
            gap = event.get("gap_m")
            if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not 0 < float(gap) <= 200.0:
                _fail(f"traffic['events'][{index}]['gap_m'] 必须在 (0, 200] 内，实际 {gap!r}")
            speed_mps = event.get("speed_mps")
            if isinstance(speed_mps, bool) or not isinstance(speed_mps, (int, float)) or not 0 < float(speed_mps) <= MAX_EVENT_SPEED_MPS:
                _fail(
                    f"traffic['events'][{index}]['speed_mps'] 必须在 (0, {MAX_EVENT_SPEED_MPS}] 内，实际 {speed_mps!r}"
                )
        if not event_types <= set(patterns):
            _fail(f"事件类型 {sorted(event_types)} 必须包含在 traffic['patterns']={patterns} 中")

        # ---- nav ----
        nav = self.nav
        if not isinstance(nav, dict):
            _fail("nav 必须是 dict")
        turns = nav.get("turns")
        if not isinstance(turns, list) or len(turns) != len(self.blocks):
            _fail(f"nav['turns'] 必须是与 blocks 等长的列表，实际 {turns!r}")
        for index, (char, turn) in enumerate(zip(self.blocks, turns)):
            if turn not in NAV_MANEUVER_LABELS:
                _fail(f"nav['turns'][{index}]={turn!r} 不在 {NAV_MANEUVER_LABELS} 中")
            if turn not in TURN_CANDIDATES[char]:
                _fail(f"nav['turns'][{index}]={turn!r} 不是 block {char!r} 的合法候选 {TURN_CANDIDATES[char]}")
        if nav.get("junction_blocks") != junction_indices(self.blocks):
            _fail(f"nav['junction_blocks']={nav.get('junction_blocks')!r} 与 junction_indices(blocks)={junction_indices(self.blocks)} 不一致")
        if nav.get("candidates") != [list(TURN_CANDIDATES[char]) for char in self.blocks]:
            _fail(f"nav['candidates']={nav.get('candidates')!r} 与 blocks 候选表不一致")
        if nav.get("source") != "block_sequence":
            _fail(f"nav['source'] 必须是 'block_sequence'，实际 {nav.get('source')!r}")

        # ---- ego ----
        ego = self.ego
        if not isinstance(ego, dict):
            _fail("ego 必须是 dict")
        spawn_lane = ego.get("spawn_lane_index")
        if spawn_lane is not None:
            if not isinstance(spawn_lane, (list, tuple)) or len(spawn_lane) != 3:
                _fail(f"ego['spawn_lane_index'] 必须是 None 或 (road_start, road_end, lane_id) 三元组，实际 {spawn_lane!r}")
            road = (spawn_lane[0], spawn_lane[1])
            if road != EGO_SPAWN_ROAD:
                _fail(
                    f"ego['spawn_lane_index'] 的 road 键必须是首块固定节点 {EGO_SPAWN_ROAD}"
                    f"（BIG 自动前置 First block），实际 {road!r}"
                )
            lane_id = spawn_lane[2]
            if isinstance(lane_id, bool) or not isinstance(lane_id, int) or not 0 <= lane_id < EGO_SPAWN_LANE_NUM:
                _fail(
                    f"ego['spawn_lane_index'][2] 必须是 [0, {EGO_SPAWN_LANE_NUM}) 内的车道号"
                    f"（0=最左，从左到右递增），实际 {lane_id!r}"
                )
        longitude = ego.get("spawn_longitude")
        if isinstance(longitude, bool) or not isinstance(longitude, (int, float)) or not 0 <= float(longitude) < 10.0:
            _fail(f"ego['spawn_longitude'] 必须在 [0, 10) 内（First block 入口段长 10 m），实际 {longitude!r}")
        lateral = ego.get("spawn_lateral")
        if isinstance(lateral, bool) or not isinstance(lateral, (int, float)) or abs(float(lateral)) > 5.0:
            _fail(f"ego['spawn_lateral'] 必须在 [-5, 5] 内（m），实际 {lateral!r}")
        velocity = ego.get("spawn_velocity")
        if isinstance(velocity, bool) or not isinstance(velocity, (int, float)) or not 0 <= float(velocity) <= MAX_EVENT_SPEED_MPS:
            _fail(f"ego['spawn_velocity'] 必须在 [0, {MAX_EVENT_SPEED_MPS}] 内（m/s），实际 {velocity!r}")

        # ---- labels 与其余字段的交叉一致性 ----
        labels = self.labels
        if not isinstance(labels, dict):
            _fail("labels 必须是 dict")
        for key in ("geometry", "traffic", "control", "maneuver", "difficulty"):
            if key not in labels:
                _fail(f"labels 缺少键 {key!r}")
        if labels["geometry"] not in self.geometry:
            _fail(f"labels['geometry']={labels['geometry']!r} 不在 geometry={self.geometry} 中")
        if labels["traffic"] != patterns:
            _fail(f"labels['traffic']={labels['traffic']!r} 与 traffic['patterns']={patterns!r} 不一致")
        if labels["control"] not in CONTROL_LABELS:
            _fail(f"labels['control']={labels['control']!r} 不在 {CONTROL_LABELS} 中")
        expected_control = (
            "none" if not event_types else
            "cut_in_cut_out" if event_types == {"cut_in", "cut_out"} else
            f"cut_{next(iter(event_types)).split('_')[1]}"
        )
        if labels["control"] != expected_control:
            _fail(f"labels['control']={labels['control']!r} 与事件类型 {sorted(event_types)} 派生值 {expected_control!r} 不一致")
        if labels["maneuver"] != turns:
            _fail(f"labels['maneuver']={labels['maneuver']!r} 与 nav['turns']={turns!r} 不一致")
        if labels["difficulty"] != self.difficulty:
            _fail(f"labels['difficulty']={labels['difficulty']!r} 与 difficulty={self.difficulty!r} 不一致")


def save_specs(specs: "list[ScenarioSpec]", path) -> None:
    """写入 JSON（先写临时文件再替换，避免半截文件被下游读到）。"""
    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "count": len(specs),
        "specs": [spec.to_dict() for spec in specs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    tmp_path.replace(path)


def load_specs(path) -> "list[ScenarioSpec]":
    """读取 JSON（支持落盘对象或裸列表）；不做 validate，校验由 validator 负责。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "specs" not in payload:
            raise ValueError(f"{path} 缺少 'specs' 字段")
        payload = payload["specs"]
    if not isinstance(payload, list):
        raise ValueError(f"{path} 内容必须是 spec 列表或含 'specs' 的对象")
    return [ScenarioSpec.from_dict(item) for item in payload]
