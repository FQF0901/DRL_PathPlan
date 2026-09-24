"""场景标签体系：几何/交通/控制/导航标签与 BIG block 字符映射。

用途
----
spec / generator / validator / labels 各线共用同一套标签常量与 block 映射，集中一处
避免多处硬编码漂移。本模块只依赖标准库，import 无副作用。

关键接口
--------
- ``GEOMETRY_LABELS`` / ``TRAFFIC_PATTERNS`` / ``CONTROL_LABELS`` /
  ``NAV_MANEUVER_LABELS`` / ``DIFFICULTIES`` / ``SPLITS``
- ``EXCLUDED_GEOMETRY_LABELS`` / ``SAMPLEABLE_GEOMETRY_LABELS``: 生成集排除的几何与可抽样几何
- ``ROUNDABOUT_MAX_DENSITY``: 含环岛场景的密度上限（追尾不可规避，见常量注释）
- ``BLOCK_CHARS``: 几何标签 -> BIG block 字符
- ``EGO_SPAWN_ROAD`` / ``EGO_SPAWN_LANE_NUM``: 自车 spawn 首块的 road 键与车道数
- ``sequence_for(geometry)``: 几何标签序列 -> map 字符串（不含自动前置的 First block）
- ``allocate_counts(n, labels, min_per_label)``: 分层配额，总数恰为 n
- ``turns_for(blocks, rng)``: 名义转向序列（与 block 字符一一对应）

MetaDrive 0.4.3 源码核对（只读，路径 ``.venv/lib/python3.10/site-packages/metadrive/``）
--------------------------------------------------------------------------------------
- ``component/algorithm/BIG.py:86``：字符串生成时 ``self._block_sequence = FirstPGBlock.ID + parameter``，
  即 First block（``component/pgblock/first_block.py:24``，ID="I"）**自动前置**。
  因此 ``blocks`` 字符串不得再包含 "I"：``BIG.py:116-117`` 会逐字符调用
  ``PGBlockDistConfig.get_block``，而 V2 分布（``blocks_prob_dist.py:22-40``）里没有 First block，
  含 "I" 会直接抛 ``ValueError``。
- 可用字符由 V2 分布的名字经 ``get_block``（``blocks_prob_dist.py:52-60``）反查 ID 得到
  （注册表见 ``utils/registry.py:10-29``）：S C r R y Y X T O U B $。
- ``component/pgblock/fork.py:28`` 与 ``fork.py:178``：InFork('f')、OutFork('F') 的
  ``_try_plug_into_previous_block`` 均直接 ``raise ValueError``，禁止使用。
- 各 block ID 出处：straight.py:16 'S'、curve.py:18 'C'、ramp.py:39 'r'、ramp.py:227 'R'、
  bottleneck.py:43 'y' (Merge)、bottleneck.py:187 'Y' (Split)、intersection.py:36 'X'、
  t_intersection.py:13 'T'、intersection.py:269 'U'、roundabout.py:16 'O'、
  bidirection.py:72 'B'、tollgate.py:17 '$'。
- 交叉口/环岛的接续 socket 由 ``BIG.py:121`` 的 ``self.np_random.choice(sockets)`` 随机决定，
  建图前不可知，所以 ``turns_for`` 给出的是"名义"转向；运行期可观测转向来自
  ``navigation_command``（``component/navigation_module/node_network_navigation.py``）。
"""

# 几何标签：每个标签对应一类"必须出现一次"的 block；顺序即 allocate_counts 的分配顺序。
GEOMETRY_LABELS = (
    "straight",         # 直道
    "curve",            # 弯道
    "ramp_in",          # 上匝道汇入主路（InRampOnStraight）
    "ramp_out",         # 下匝道驶离主路（OutRampOnStraight）
    "merge",            # 车道数减少的汇入（Bottleneck Merge）
    "split",            # 车道数增加的分流（Bottleneck Split）
    "intersection",     # 十字交叉口
    "t_intersection",   # T 字交叉口
    "roundabout",       # 环岛
    "uturn",            # 带掉头口的交叉口
    "bidirection",      # 双向路段（已排除出生成集：S→B 接缝不可通行，见 EXCLUDED_GEOMETRY_LABELS）
    "tollgate",         # 收费站
)

# 生成集排除的几何标签及原因（标签常量保留，供未来上游修复后恢复）：
# - "bidirection"（B）：S→B 接缝几何不可通行——前一 S 块 lane-0 的左黄线与 B 块正向车道的
#   右黄线在 x 向仅隔 ~2 m，而自车底盘长 4.5 m，任何连续路径都会同时跨越两条黄线，触发
#   MetaDrive on_yellow_continuous_line -> out_of_road（手工理想路径同样失败；L2 基线诊断
#   off_road_rate=1.0）。因此 generator 只从 SAMPLEABLE_GEOMETRY_LABELS 抽样，生成的 spec
#   （geometry / blocks）永不含 B。
EXCLUDED_GEOMETRY_LABELS = ("bidirection",)

# 可抽样几何标签：allocate_counts / 模板抽样的取值域 = GEOMETRY_LABELS 去掉排除项。
SAMPLEABLE_GEOMETRY_LABELS = tuple(
    label for label in GEOMETRY_LABELS if label not in EXCLUDED_GEOMETRY_LABELS
)

# 含 roundabout 的场景密度上限：环岛内层 MVehicle 以 8.33 m/s 逼近、而自车环岛限速 6.87 m/s，
# MetaDrive 的 IDM 只对同车道前车让行 -> 环岛车流导致的追尾不可规避。生成器把含 roundabout
# 的几何密度钳到 <= 该值，并在 spec.traffic["density_cap"] 记录（spec.validate 对账）。
ROUNDABOUT_MAX_DENSITY = 0.05

# 交通形态标签（spec.traffic["patterns"] 的取值域）。
TRAFFIC_PATTERNS = (
    "cut_in",           # 脚本化切入
    "cut_out",          # 脚本化切出
    "crowded",          # 高密度
    "lane_change",      # 邻车变道
    "zipper_merge",     # 匝道拉链式汇入（需 merge 几何）
)

# 控制标签：由脚本化事件派生（cut-in / cut-out 是否安装）。
CONTROL_LABELS = (
    "none",
    "cut_in",
    "cut_out",
    "cut_in_cut_out",
)

# 导航机动标签（spec.nav["turns"] 的取值域）。
NAV_MANEUVER_LABELS = (
    "straight",
    "left",
    "right",
    "roundabout_exit",
    "uturn",
)

DIFFICULTIES = ("easy", "medium", "hard")
SPLITS = ("train", "val")

# 几何标签 -> BIG block 字符（见模块 docstring 的源码核对）。
BLOCK_CHARS = {
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

# 反查表：block 字符 -> 几何标签。
BLOCK_CHARS_INV = {char: label for label, char in BLOCK_CHARS.items()}

# 禁止出现在 blocks 字符串里的字符：
# - "I"：BIG 会自动前置 First block（BIG.py:86），显式写出会 get_block 失败；
# - "f"/"F"：两个 Fork block 均直接 raise ValueError（fork.py:28/178）。
FORBIDDEN_BLOCK_CHARS = ("I", "f", "F")

# 允许的 block 字符（顺序按 GEOMETRY_LABELS）。
VALID_BLOCK_CHARS = tuple(dict.fromkeys(BLOCK_CHARS.values()))

# 小地图规模（不含隐式 First block），与 config/env.yaml blocks_per_map 一致。
SEQUENCE_MIN_BLOCKS = 3
SEQUENCE_MAX_BLOCKS = 5

# 自车 spawn 车道模型（spec.ego["spawn_lane_index"] 的取值域；generator 按此选道）：
# - BIG 自动前置 FirstPGBlock（BIG.py:86），其节点恒定（first_block.py:26-27）：
#   NODE_1=">" -> NODE_2=">>"，因此 spawn_lane_index 的 road 键与 block 序列无关；
# - 首块车道数取 METADRIVE_DEFAULT_CONFIG["map_config"][LANE_NUM]=3
#   （envs/metadrive_env.py:34）；L2 build_env 未覆盖 map_config，故实际恒为 3；
# - 车道号 0..lane_num-1 从左到右递增（create_pg_block_utils.py:68-186 把最左车道放 id 0），
#   与 L1b ``behaviors._resolve_side`` 的 left=id-1 / right=id+1 一致。
EGO_SPAWN_ROAD = (">", ">>")
EGO_SPAWN_LANE_NUM = 3

# 具备多方向选择的 block：其接续方向由 BIG 随机 socket 决定（BIG.py:121）。
JUNCTION_CHARS = ("X", "T", "U", "O")

# block 字符 -> 名义转向候选。单 socket 的 block 只能"直行通过"（含弯道/匝道）；
# 交叉口按实际存在的三个方向给候选，环岛统一记 roundabout_exit（出口方向由 socket 决定）。
TURN_CANDIDATES = {
    "S": ("straight",),
    "C": ("straight",),
    "r": ("straight",),
    "R": ("straight",),
    "y": ("straight",),
    "Y": ("straight",),
    "B": ("straight",),
    "$": ("straight",),
    "X": ("straight", "left", "right"),
    "T": ("straight", "left", "right"),
    "U": ("straight", "left", "right", "uturn"),
    "O": ("roundabout_exit",),
}


def sequence_for(geometry: list[str]) -> str:
    """把几何标签序列映射为 ``map=`` 可直接使用的 block 字符串。

    为什么不含 "I"：BIG 会自动前置 First block（BIG.py:86），调用方只需给出显式 block。
    非法标签或禁止字符（I/f/F）直接抛 ValueError，避免把坏序列带到建图阶段。
    """
    if not isinstance(geometry, (list, tuple)) or not geometry:
        raise ValueError(f"geometry 必须是非空序列，实际: {geometry!r}")
    chars = []
    for label in geometry:
        if label not in BLOCK_CHARS:
            raise ValueError(f"未知几何标签 {label!r}；合法标签: {GEOMETRY_LABELS}")
        chars.append(BLOCK_CHARS[label])
    blocks = "".join(chars)
    for char in blocks:
        if char in FORBIDDEN_BLOCK_CHARS:
            raise ValueError(f"blocks 含禁止字符 {char!r}（First block 自动前置，Fork 块不可用）")
    return blocks


def allocate_counts(n: int, labels: "list[str] | tuple[str, ...]", min_per_label: int) -> dict[str, int]:
    """按标签均分 n 个配额，保证每类 >= min_per_label，且总数恰为 n。

    为什么用"先保底再均摊"：几何覆盖是硬约束（每个标签 >= min_per_label），
    在满足覆盖后再尽量均匀；余数按标签顺序逐一分给前 ``extra`` 类，结果确定可复现。
    """
    labels = tuple(labels)
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n 必须是非负整数，实际: {n!r}")
    if min_per_label < 0:
        raise ValueError(f"min_per_label 必须非负，实际: {min_per_label!r}")
    if not labels:
        if n:
            raise ValueError(f"labels 为空但 n={n}")
        return {}
    if n < min_per_label * len(labels):
        raise ValueError(
            f"n={n} 不足以让 {len(labels)} 个标签各 >= {min_per_label}（至少需要 {min_per_label * len(labels)}）"
        )
    counts = {label: min_per_label for label in labels}
    share, extra = divmod(n - min_per_label * len(labels), len(labels))
    for index, label in enumerate(labels):
        counts[label] += share + (1 if index < extra else 0)
    return counts


def turns_for(blocks: str, rng) -> list[str]:
    """由 block 序列推导名义转向序列（长度与 blocks 相同，逐 block 对齐）。

    名义而非精确：交叉口的接续 socket 由 BIG 内部随机选取（BIG.py:121），建图前无法预知
    实际转向；运行期真正可观测的转向来自 ``navigation_command``。spec 里的 turns 用于
    场景标注与 validator 对账，不作为训练标签真值。
    """
    turns = []
    for char in blocks:
        candidates = TURN_CANDIDATES.get(char)
        if candidates is None:
            raise ValueError(f"blocks 含非法字符 {char!r}")
        turns.append(rng.choice(candidates))
    return turns


def junction_indices(blocks: str) -> list[int]:
    """返回多方向 block 的 1-based 下标（与 spec.nav["junction_blocks"] 对齐）。"""
    return [index for index, char in enumerate(blocks, start=1) if char in JUNCTION_CHARS]
