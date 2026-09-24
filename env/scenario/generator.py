"""场景 spec 生成器：几何分层抽样 + 难度分层 + 事件窗口 + 名义导航转向。

关键接口
--------
``build_specs(n_train, n_val, train_seeds, val_seeds, rng_seed=0) -> (train, val)``

设计要点（为什么这么做）
----------------------
- **几何覆盖是硬约束**：``allocate_counts`` 先给每个可抽样几何标签保底 ``min_per_label`` 再均摊，
  保证全部可抽样几何（11 类）在每个 split 都有足够样本（contract §3-L1a）。
  生成集排除 ``bidirection``（S→B 接缝不可通行，见 ``taxonomy.EXCLUDED_GEOMETRY_LABELS``），
  生成的 spec 永不含 ``B``。
- **环岛密度上限**：含 ``roundabout`` 的几何把密度钳到 ``taxonomy.ROUNDABOUT_MAX_DENSITY``
  （环岛车流追尾不可规避），上限记录在 ``traffic["density_cap"]``；派生难度随之下降是预期行为。
- **难度派生而非随机**：先选目标难度档（round-robin 保证均衡），再从该档的密度/间隙区间采样，
  最后由 ``_derive_difficulty`` 从密度、最小事件间隙、交叉口数量重新派生难度——
  档位与派生值一致（见下方区间设计），标签因此"可解释"。
- **内容只由 (rng_seed, seed) 决定**：与总数 n 无关，切片与全量可独立生成且同 seed 内容稳定，
  便于复现与失败定位。
- **自车 spawn 车道显式化**：``_select_spawn_lane`` 按事件请求侧在首块（恒 3 车道，
  见 ``taxonomy.EGO_SPAWN_LANE_NUM``）选中间车道，保证 cut_in/cut_out 请求侧的邻车道存在；
  车道数不足导致请求侧不可达（如 2 车道要外侧）时改选可行车道并就地改写事件 ``side``，
  避免 L1b 事件车降级到对侧（``scripted_event_degraded``）。选道不消耗 rng，
  同 seed 仍输出完全相同的 spec。
- **名义转向**：显式 block 序列里多 socket block 的接续 socket 由 BIG 内部随机选取
  （BIG.py:121），建图前不可知；spec.nav["turns"] 记录名义意图，运行期以
  ``navigation_command`` 为准（validator 对账）。

难度区间（改这里时同步改 ``_derive_difficulty``）
-----------------------------------------------
========  ==============  ==================  ==========================
难度      密度            事件 gap (m)        其它
========  ==============  ==================  ==========================
easy      [0.02, 0.06]    [18, 30]（仅 cut_out，40% 概率）  无 cut_in
medium    [0.08, 0.14]    [12, 18]（必有 1 个事件）         —
hard      [0.16, 0.28]    [6, 12]（必有 cut_in，50% 加 1 个） —
========  ==============  ==================  ==========================
派生分数：密度 0/1/2 + 间隙 0/1/2 + 双交叉口 0/1 → <=1 easy / <=3 medium / else hard。
注：含 ``roundabout`` 的几何密度被钳到 ``ROUNDABOUT_MAX_DENSITY``（<=0.05），其派生难度可能
低于目标档（如 hard 目标降为 medium），这是"难度以派生值为准"的预期结果。
"""

import random

from .spec import ScenarioSpec
from .taxonomy import (
    BLOCK_CHARS,
    DIFFICULTIES,
    EGO_SPAWN_LANE_NUM,
    EGO_SPAWN_ROAD,
    EXCLUDED_GEOMETRY_LABELS,
    GEOMETRY_LABELS,
    JUNCTION_CHARS,
    ROUNDABOUT_MAX_DENSITY,
    SAMPLEABLE_GEOMETRY_LABELS,
    TURN_CANDIDATES,
    allocate_counts,
    junction_indices,
    sequence_for,
    turns_for,
)

# 每个场景的显式 block 数（不含隐式 First block），与 config/env.yaml blocks_per_map 一致。
BLOCKS_PER_MAP = (3, 5)

# 各 block 字符的限速（m/s）：高速直道 ~13.9，城市/弯道 ~8.3，环岛/掉头更低。
SPEED_LIMIT_MPS_BY_CHAR = {
    "S": 13.9,
    "C": 8.3,
    "r": 8.3,
    "R": 8.3,
    "y": 11.1,
    "Y": 11.1,
    "X": 8.3,
    "T": 8.3,
    "O": 6.9,
    "U": 5.6,
    "B": 8.3,
    "$": 5.6,
}

DENSITY_RANGE = {
    "easy": (0.02, 0.06),
    "medium": (0.08, 0.14),
    "hard": (0.16, 0.28),
}
EVENT_GAP_M = {
    "easy": (18.0, 30.0),
    "medium": (12.0, 18.0),
    "hard": (6.0, 12.0),
}
EGO_SPEED_MPS = {
    "easy": 6.0,
    "medium": 8.3,
    "hard": 11.1,
}
_EVENT_TRIGGER_STEPS = {
    "easy": (30, 90),
    "medium": (40, 110),
    "hard": (30, 90),
}
_EVENT_DURATION_STEPS = {
    "easy": (40, 70),
    "medium": (30, 60),
    "hard": (25, 50),
}
_CROWDED_DENSITY = 0.12      # >= 该密度才标 crowded
_LANE_CHANGE_PROB = 0.5      # 非 easy 场景追加 lane_change 形态的概率
_EASY_CUT_OUT_PROB = 0.4     # easy 场景可选 cut_out（gap 大，不破坏 easy 语义）

# 生成器内部自检用：所有字符都要有限速与转向候选。
assert set(SPEED_LIMIT_MPS_BY_CHAR) == set(BLOCK_CHARS.values()), "限速表必须覆盖全部 block 字符"

# 每个几何标签的候选 block 序列模板（标签级，不含 First block）。
# 顺序不敏感：_sample_geometry 会先按 (交叉口数量, 长度) 排序再按难度分位选择，
# 以确保 easy 选简单地图、hard 选复杂地图。模板里禁止出现 'f'/'F'（Fork 块 raise）。
_GEOMETRY_TEMPLATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "straight": (
        ("straight", "straight", "straight"),
        ("straight", "straight", "straight", "straight"),
        ("curve", "straight", "straight"),
        ("curve", "straight", "straight", "straight", "straight"),
    ),
    "curve": (
        ("straight", "curve", "straight"),
        ("curve", "straight", "curve"),
        ("straight", "curve", "straight", "curve"),
        ("straight", "curve", "straight", "curve", "straight"),
    ),
    "ramp_in": (
        ("straight", "straight", "ramp_in"),
        ("curve", "straight", "ramp_in"),
        ("straight", "ramp_in", "straight"),
        ("straight", "straight", "ramp_in", "straight"),
    ),
    "ramp_out": (
        ("straight", "straight", "ramp_out"),
        ("curve", "ramp_out", "straight"),
        ("straight", "ramp_out", "straight"),
    ),
    "merge": (
        ("straight", "straight", "merge"),
        ("curve", "straight", "merge"),
        ("straight", "merge", "straight"),
    ),
    "split": (
        ("straight", "split", "straight"),
        ("straight", "straight", "split"),
        ("curve", "split", "straight"),
    ),
    "intersection": (
        ("straight", "intersection", "straight"),
        ("curve", "intersection", "straight"),
        ("straight", "intersection", "curve"),
        ("straight", "curve", "intersection", "ramp_out", "roundabout"),
    ),
    "t_intersection": (
        ("straight", "t_intersection", "straight"),
        ("curve", "t_intersection", "straight"),
        ("straight", "t_intersection", "curve"),
    ),
    "roundabout": (
        ("straight", "roundabout", "straight"),
        ("curve", "roundabout", "straight"),
        ("straight", "roundabout", "curve"),
        ("straight", "curve", "intersection", "ramp_out", "roundabout"),
    ),
    "uturn": (
        ("straight", "uturn", "straight"),
        ("curve", "uturn", "straight"),
        ("straight", "uturn", "curve"),
    ),
    "bidirection": (  # 已排除出生成集（见 taxonomy.EXCLUDED_GEOMETRY_LABELS）；模板保留供未来修复后恢复
        ("straight", "bidirection", "straight"),
        ("straight", "straight", "bidirection"),
        ("curve", "bidirection", "straight"),
    ),
    "tollgate": (
        ("straight", "straight", "tollgate"),
        ("curve", "straight", "tollgate"),
        ("straight", "tollgate", "straight"),
    ),
}


def _validate_templates() -> None:
    """import 期自检模板表：覆盖全部标签、长度合法、主标签出现、字符合法。

    额外保证可抽样模板不含被排除几何的 block 字符（生成器因此永不会产出 ``B``）。
    """
    for label in GEOMETRY_LABELS:
        templates = _GEOMETRY_TEMPLATES.get(label)
        if not templates:
            raise ValueError(f"几何标签 {label!r} 缺少模板")
        for template in templates:
            if not (BLOCKS_PER_MAP[0] <= len(template) <= BLOCKS_PER_MAP[1]):
                raise ValueError(f"模板 {template} 长度超出 {BLOCKS_PER_MAP}")
            if label not in template:
                raise ValueError(f"模板 {template} 不含其主标签 {label!r}")
            sequence_for(list(template))  # 顺带校验标签与字符合法
    excluded_chars = {BLOCK_CHARS[label] for label in EXCLUDED_GEOMETRY_LABELS}
    for label in SAMPLEABLE_GEOMETRY_LABELS:
        for template in _GEOMETRY_TEMPLATES[label]:
            bad = sorted(char for char in sequence_for(list(template)) if char in excluded_chars)
            if bad:
                raise ValueError(f"可抽样模板 {template} 含被排除字符 {bad}；生成集不得含 {sorted(excluded_chars)}")


_validate_templates()


def _combine_seed(rng_seed: int, seed: int) -> int:
    """把 (rng_seed, seed) 映射为单个整数种子：整数运算，跨进程可复现（不用 str hash）。"""
    return (int(rng_seed) << 32) ^ int(seed)


def _resolve_seeds(name: str, seed_range, n: int) -> list[int]:
    """把 (start, stop) 解析为 n 个连续 seed；区间不足或非法直接报错（宁可失败不可泄漏）。"""
    if not isinstance(seed_range, (tuple, list)) or len(seed_range) != 2:
        raise ValueError(f"{name} seed 区间必须是 (start, stop)，实际 {seed_range!r}")
    start, stop = int(seed_range[0]), int(seed_range[1])
    if start < 0:
        raise ValueError(f"{name} seed 起点必须非负，实际 {start}")
    if n < 0:
        raise ValueError(f"{name} 条数必须非负，实际 {n}")
    if stop - start < n:
        raise ValueError(f"{name} seed 区间 [{start}, {stop}) 容不下 {n} 条")
    return [start + index for index in range(n)]


def _sample_geometry(rng: random.Random, primary: str, difficulty: str) -> tuple[str, ...]:
    """按难度分位选择模板：easy 取最简、medium 取中段、hard 取最复杂（含多交叉口组合）。"""
    templates = sorted(
        _GEOMETRY_TEMPLATES[primary],
        key=lambda template: (
            sum(1 for label in template if BLOCK_CHARS[label] in JUNCTION_CHARS),
            len(template),
        ),
    )
    count = len(templates)
    if difficulty == "easy":
        low, high = 0, max(1, count // 3)
    elif difficulty == "medium":
        low, high = count // 3, max(count // 3 + 1, (2 * count) // 3)
    else:
        low, high = (2 * count) // 3, count
    return templates[rng.randrange(low, high)]


def _make_event(rng: random.Random, kind: str, difficulty: str, trigger_step: int, speed_mps: float) -> dict:
    """构造单个脚本化事件窗口（步数按 10 Hz env step；速度 m/s，与 spawn_velocity 同单位）。"""
    return {
        "type": kind,
        "side": rng.choice(("left", "right")),
        "trigger_step": int(trigger_step),
        "duration_steps": int(rng.randint(*_EVENT_DURATION_STEPS[difficulty])),
        "gap_m": round(rng.uniform(*EVENT_GAP_M[difficulty]), 1),
        "speed_mps": round(speed_mps, 1),
    }


def _sample_events(rng: random.Random, difficulty: str, ego_speed_mps: float) -> list[dict]:
    """按难度生成事件窗口：easy 仅可选 cut_out；medium/hard 必有事件，hard 更近更快。"""
    events: list[dict] = []
    trigger = rng.randint(*_EVENT_TRIGGER_STEPS[difficulty])
    if difficulty == "easy":
        if rng.random() < _EASY_CUT_OUT_PROB:
            events.append(_make_event(rng, "cut_out", difficulty, trigger, ego_speed_mps * 1.1))
    elif difficulty == "medium":
        kind = "cut_in" if rng.random() < 0.5 else "cut_out"
        events.append(_make_event(rng, kind, difficulty, trigger, ego_speed_mps * rng.uniform(0.9, 1.2)))
    else:
        events.append(_make_event(rng, "cut_in", difficulty, trigger, ego_speed_mps * rng.uniform(0.9, 1.15)))
        if rng.random() < 0.5:
            later = trigger + rng.randint(60, 120)
            kind = rng.choice(("cut_in", "cut_out"))
            events.append(_make_event(rng, kind, difficulty, later, ego_speed_mps * rng.uniform(0.9, 1.15)))
    return events


def _select_spawn_lane(events: list[dict], lane_num: int = EGO_SPAWN_LANE_NUM) -> int:
    """选择自车 spawn 车道号（0=最左，从左到右递增），保证事件请求侧存在邻车道。

    纯函数、不消耗 rng（同 seed 输出不变）：
    - ``lane_num >= 3``：取中间车道——左右邻道都存在，任意请求侧都无需降级；
    - ``lane_num == 2``：左右不能兼顾，按请求侧取唯一可行车道（left -> 1，right -> 0）；
      两侧同时请求时先满足首个事件的 side，另一侧由 ``_align_event_sides`` 改写；
    - ``lane_num <= 1``：只有唯一车道且无侧邻道，返回 0（降级不可避免，见 _align_event_sides）。
    """
    if lane_num >= 3:
        return lane_num // 2
    if lane_num == 2:
        sides = {event["side"] for event in events}
        if sides == {"left"}:
            return 1
        if sides == {"left", "right"}:
            first = next(event["side"] for event in events if event["side"] in ("left", "right"))
            return 1 if first == "left" else 0
    return 0


def _align_event_sides(events: list[dict], lane_index: int, lane_num: int = EGO_SPAWN_LANE_NUM) -> None:
    """就地把请求了不可达侧的事件 ``side`` 改写到当前车道可用的侧。

    车道号 0 无左邻道、``lane_num-1`` 无右邻道；两侧都不可用（``lane_num <= 1``，当前配置
    不会出现）时保留原 side，此时 L1b 事件车只能降级（半车道偏移），属不可避免的残留。
    调整结果直接落在 spec.traffic["events"][*]["side"]，即在 spec 中可审计。
    """
    available = set()
    if lane_index - 1 >= 0:
        available.add("left")
    if lane_index + 1 <= lane_num - 1:
        available.add("right")
    if not available:
        return
    fallback = "left" if "left" in available else "right"
    for event in events:
        if event["side"] not in available:
            event["side"] = fallback


def _derive_difficulty(density: float, events: list[dict], n_junctions: int) -> str:
    """由密度/最小事件间隙/交叉口数派生难度，必须与采样档一致（见模块 docstring 表格）。"""
    density_score = 0 if density < 0.08 else (1 if density < 0.16 else 2)
    if not events:
        gap_score = 0
    else:
        min_gap = min(event["gap_m"] for event in events)
        gap_score = 1 if min_gap >= 12.0 else 2
    junction_score = 1 if n_junctions >= 2 else 0
    score = density_score + gap_score + junction_score
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def _derive_patterns(
    rng: random.Random,
    geometry: tuple[str, ...],
    density: float,
    events: list[dict],
    difficulty: str,
) -> list[str]:
    """派生交通形态标签：事件类型必含；高密度记 crowded；merge 几何记 zipper_merge。"""
    patterns: list[str] = []
    for event in events:
        if event["type"] not in patterns:
            patterns.append(event["type"])
    if density >= _CROWDED_DENSITY:
        patterns.append("crowded")
    if difficulty != "easy" and rng.random() < _LANE_CHANGE_PROB:
        patterns.append("lane_change")
    if "merge" in geometry and difficulty != "easy":
        patterns.append("zipper_merge")
    return patterns


def _control_label(events: list[dict]) -> str:
    """由事件类型派生控制标签（与 taxonomy.CONTROL_LABELS 对齐）。"""
    types = {event["type"] for event in events}
    if not types:
        return "none"
    if types == {"cut_in"}:
        return "cut_in"
    if types == {"cut_out"}:
        return "cut_out"
    return "cut_in_cut_out"


def _make_spec(split: str, spec_id: int, seed: int, primary: str, target_difficulty: str, rng_seed: int) -> ScenarioSpec:
    """构造单条 spec；所有随机量来自 (rng_seed, seed)，与总数 n 无关。"""
    if primary not in SAMPLEABLE_GEOMETRY_LABELS:
        raise ValueError(f"几何标签 {primary!r} 已排除出生成集（见 taxonomy.EXCLUDED_GEOMETRY_LABELS）")
    rng = random.Random(_combine_seed(rng_seed, seed))

    density = round(rng.uniform(*DENSITY_RANGE[target_difficulty]), 3)
    geometry = _sample_geometry(rng, primary, target_difficulty)
    # 环岛密度上限：环岛车流追尾不可规避（见 taxonomy.ROUNDABOUT_MAX_DENSITY），钳位后记入 traffic。
    if "roundabout" in geometry and density > ROUNDABOUT_MAX_DENSITY:
        density = ROUNDABOUT_MAX_DENSITY
    blocks = sequence_for(list(geometry))
    n_junctions = len(junction_indices(blocks))
    ego_speed = EGO_SPEED_MPS[target_difficulty]
    events = _sample_events(rng, target_difficulty, ego_speed)
    # 显式 spawn 车道：保证每个事件的请求侧都有邻车道；不可达侧就地改写（不消耗 rng）。
    spawn_lane = _select_spawn_lane(events)
    _align_event_sides(events, spawn_lane)
    # 难度以派生值为准：区间设计保证与 target 一致，若未来区间被改动则这里如实反映。
    difficulty = _derive_difficulty(density, events, n_junctions)
    patterns = _derive_patterns(rng, geometry, density, events, difficulty)
    turns = turns_for(blocks, rng)

    # 限速按 block 字符取，场景级取最小值（整条路线统一用最保守限速，交由 L2 建图后写入）。
    by_char = {char: SPEED_LIMIT_MPS_BY_CHAR[char] for char in dict.fromkeys(blocks)}
    traffic = {
        "density": density,
        "patterns": patterns,
        "events": events,
        "seed": seed,
        "random_traffic": False,
    }
    if "roundabout" in geometry:  # 记录环岛密度上限（密度已在上方钳位）
        traffic["density_cap"] = ROUNDABOUT_MAX_DENSITY
    spec = ScenarioSpec(
        id=spec_id,
        seed=seed,
        split=split,
        blocks=blocks,
        geometry=list(geometry),
        traffic=traffic,
        limits={
            "speed_limit_mps": min(by_char.values()),
            "by_char": by_char,
        },
        nav={
            "turns": turns,
            "junction_blocks": junction_indices(blocks),
            "candidates": [list(TURN_CANDIDATES[char]) for char in blocks],
            "source": "block_sequence",
        },
        ego={
            "spawn_lane_index": [EGO_SPAWN_ROAD[0], EGO_SPAWN_ROAD[1], spawn_lane],
            "spawn_longitude": round(rng.uniform(4.0, 8.0), 2),
            "spawn_lateral": 0.0,
            "spawn_velocity": round(ego_speed, 2),
        },
        difficulty=difficulty,
        labels={
            "geometry": primary,
            "traffic": patterns,
            "control": _control_label(events),
            "maneuver": turns,
            "difficulty": difficulty,
        },
    )
    spec.validate()
    return spec


def _build_split(split: str, n: int, seed_range, rng_seed: int) -> list[ScenarioSpec]:
    """生成一个 split：先按几何标签分配配额并打散，再逐条构造 spec。"""
    if n < 0:
        raise ValueError(f"{split} 条数必须非负，实际 {n}")
    if n == 0:
        return []
    seeds = _resolve_seeds(split, seed_range, n)
    # 小样本无法覆盖全部可抽样几何时退化为纯均摊（min_per_label=0）；否则每类至少 min(50, n/(2L)) 条。
    min_per_label = 0 if n < len(SAMPLEABLE_GEOMETRY_LABELS) else max(1, min(50, n // (2 * len(SAMPLEABLE_GEOMETRY_LABELS))))
    counts = allocate_counts(n, SAMPLEABLE_GEOMETRY_LABELS, min_per_label)
    slots = [label for label in SAMPLEABLE_GEOMETRY_LABELS for _ in range(counts[label])]
    # 打散槽位：否则同标签连续出现，头 N 条会全是 straight，切片抽样也失衡。
    shuffle_rng = random.Random(_combine_seed(rng_seed, 0x51ED if split == "train" else 0x7A1D))
    shuffle_rng.shuffle(slots)

    specs = []
    for spec_id, (seed, primary) in enumerate(zip(seeds, slots)):
        difficulty = DIFFICULTIES[spec_id % len(DIFFICULTIES)]
        specs.append(_make_spec(split, spec_id, seed, primary, difficulty, rng_seed))
    return specs


def build_specs(
    n_train: int,
    n_val: int,
    train_seeds: tuple[int, int],
    val_seeds: tuple[int, int],
    rng_seed: int = 0,
) -> tuple[list[ScenarioSpec], list[ScenarioSpec]]:
    """生成 (train, val) 两个 split 的 spec 列表。

    - ``train_seeds`` / ``val_seeds``：``(start, stop)`` 左闭右开区间，必须互不重叠
      （train/val seed 泄漏会直接导致评估失真，因此这里强校验）。
    - 每个 split 内每类几何 >= min_per_label，难度档 round-robin 均衡。
    """
    train_seeds = tuple(int(v) for v in train_seeds)
    val_seeds = tuple(int(v) for v in val_seeds)
    if n_train > 0 and n_val > 0 and train_seeds[0] < val_seeds[1] and val_seeds[0] < train_seeds[1]:
        raise ValueError(f"train {train_seeds} 与 val {val_seeds} seed 区间重叠，拒绝生成（防泄漏）")
    train_specs = _build_split("train", n_train, train_seeds, rng_seed)
    val_specs = _build_split("val", n_val, val_seeds, rng_seed)
    return train_specs, val_specs
