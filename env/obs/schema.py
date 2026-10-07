"""obs schema 的**机器可读清单**（键 / 形状 / dtype / 语义 / 单位）。

单一出处：``tools/collect_expert.py`` 把 :func:`schema_manifest` 的返回值写进
``expert_bc.meta.json::schema``，下游（net / trainer lane）按契约开发时直接读它，
不再从代码或 README 猜形状。形状里的槽位数/帧数由调用方传入（默认 16 / 6）。

为什么把 scope / 槽位策略 / others 布局也写进来：v2 的语义变化（OD 槽位 = track id、
``od_mask`` vs ``od_presence``、``hist_valid`` 按真实帧、road_class 顺序）都不是形状能表达的，
必须成文，否则下游很容易按 v1 语义误用。

v3（2026-10-01，A4 nav 修正）：新增 ``world`` 段 —— ``ego_world``（t0 世界系位姿）与
``route_world``（世界系路线折线 + 逐点 mask），供 rollout / 教师强制按新位姿重算 nav。

v4（2026-10-02，P1-A 静态障碍可观测性）：``others`` 新增 ``static`` 段（present +
gap_norm + 相对车道 one-hot）——收费站岗亭是 BaseBuilding，不在 OD/LD 里，必须扫描
建筑补足（见 :mod:`env.obs.static` 与 triage 签名 S1–S4）。

v5（2026-10-05，结构迭代 A）：LD offset 改远场 {20,40,60,80} m；新增 ``lane``（当前车道
块）与 ``ttc``（per-OD-slot TTC 上下文 token）。

v6（2026-10-07，v8 结构重构）：删除 ``lane``/``ttc`` 通道（v5 结构迭代 A 的上下文 token
在 v8 中不再使用）；LD offset 改为 {0,20,40,60,80} m——0 m 点 = ego 投影点（近场横向锚，
补偿 lane 删除），远场保留 20..80 m 前视。OD 槽序/字段/排序策略不动。
"""

from __future__ import annotations

from typing import Any

from env.obs.ego import EGO_DIM
from env.obs.ld import LDChannel, LD_OFFSETS_M
from env.obs.nav import NUM_CHECKPOINTS, NUM_COMMANDS
from env.obs.others import (
    NAV_DIM,
    OTHERS_HEAD_DIM,
    SPEED_LIMIT_UNSET,
    STATIC_OFFSET,
    road_class_labels,
)
from env.obs.signal import SIGNAL_DIM
from env.obs.static import (
    STATIC_DIM,
    STATIC_LANE_SPAN,
    STATIC_LAT_MARGIN_M,
    STATIC_REL_BUCKETS,
    STATIC_SCAN_RANGE_M,
)
from env.obs.world import EGO_WORLD_DIM, ROUTE_WORLD_DIM, ROUTE_WORLD_MAX_POINTS

__all__ = [
    "schema_manifest",
    "OD_SLOT_POLICY",
    "OTHERS_LAYOUT",
    "STATIC_LAYOUT",
    "WORLD_LAYOUT",
    "LD_LAYOUT",
]

LD_DIM = LDChannel.feature_dim
#: OD 特征维（见 env/obs/od.py）
OD_DIM = 9
#: nav 特征维（2 checkpoint × (x,y) + 6 one-hot + route_completion）
NAV_FEATURE_DIM = 2 * NUM_CHECKPOINTS + NUM_COMMANDS + 1

OD_FEATURE_NAMES = ("dx", "dy", "vx", "vy", "cos_theta", "sin_theta", "length", "width", "type_id")
LD_FEATURE_NAMES = ("dx", "dy", "heading_rel", "curvature", "speed_limit", "left_line_type_id", "right_line_type_id")
EGO_FEATURE_NAMES = ("v", "a_long", "a_lat", "yaw_rate", "steer", "curvature", "prev_ds", "prev_dtheta")
NAV_FEATURE_NAMES = (
    "checkpoint0_dx",
    "checkpoint0_dy",
    "checkpoint1_dx",
    "checkpoint1_dy",
    "cmd_forward",
    "cmd_left",
    "cmd_right",
    "cmd_reserved0",
    "cmd_reserved1",
    "cmd_reserved2",
    "route_completion",
)
SIGNAL_FEATURE_NAMES = ("green", "yellow", "red", "unknown")

#: OD 固定槽位策略（v2）
OD_SLOT_POLICY: dict[str, Any] = {
    "slot_identity": "track id（每 episode id->slot 表；reset() 清空）",
    "allocation": "新对象取第一个空槽；新对象按 (min(TTC, 5s), 距离) 紧迫度排序占位",
    "eviction": "槽满时紧迫度最高的新对象驱逐'最长未出现且已出盒'的槽（未观测、连续缺失最久；并列取最小下标）",
    "presence": "对象出盒 → presence=0（槽保留，特征=最近一次盒内观测）；出盒持续 > release_after_s(默认 1.0s) → 释放（id=-1、特征清零）",
    "mask": "od_mask=1 ⇔ 槽已分配（od_id>=0）——槽位级'本帧有效'；不是'对象在盒内'",
    "presence_flag": "od_presence=1 ⇔ 对象本帧在盒内被观测（特征新鲜）；下游几何使用前必须检查 presence",
    "scope": "盒式 scope 前 150 / 后 50 / 左 25 / 右 25 m（自车系 x 前 / y 左）；v2 由前 100 扩到 150",
    "id_space": "episode 内稳定整数（从 1 递增）；-1 = 空槽；跨 episode 不复用序号",
    "time_axis": "释放计时用 env.episode_step 的真实时间轴（采集每 5 env step build 一次）",
}

#: others 通道布局（schema v4 规范上下文输入）
OTHERS_LAYOUT: dict[str, Any] = {
    "shape": [1, OTHERS_HEAD_DIM + len(road_class_labels())],
    "segments": {
        "nav": {"dims": [0, NAV_DIM], "unit": "m / one-hot / [0,1]", "source": "env.obs.nav.NavChannel 同口径"},
        "speed_limit": {"dims": [NAV_DIM, NAV_DIM + 1], "unit": "[0,1]（clamp(m/s,0,norm)/norm, norm=30 m/s）"},
        "signal": {"dims": [NAV_DIM + 1, STATIC_OFFSET], "unit": "one-hot [green,yellow,red,unknown]，本项目恒 [0,0,0,1]"},
        "static": {
            "dims": [STATIC_OFFSET, OTHERS_HEAD_DIM],
            "unit": "0/1 + [0,1] + one-hot（见 static_layout）",
            "source": "env.obs.static：BaseBuilding 走廊扫描（建筑/岗亭）",
        },
        "road_class": {
            "dims": [OTHERS_HEAD_DIM, OTHERS_HEAD_DIM + len(road_class_labels())],
            "unit": "one-hot（ego 当前 lane 的 block 几何类别；未知/缺失 → 全 0）",
        },
    },
    "road_class_order": list(road_class_labels()),
    "road_class_source": "behaviors.map_info(env).lane_block[ego.lane_index]（与 labels.compute_step_labels 同源）",
    "speed_limit_unset": SPEED_LIMIT_UNSET,
    "speed_limit_unset_semantics": ">= 1000（MetaDrive 未设置）→ 0.0",
    "mask": "others_mask=1 ⇔ ego 存在；各分量有确定性回退（nav 0 / 限速 0 / 无灯 / 无静态障碍 / road_class 全 0）",
    "legacy": "旧 nav/signal 键保留以兼容，但 others 是规范输入",
}


#: others 的 static 段布局（v4，P1-A 静态障碍可观测性）
STATIC_LAYOUT: dict[str, Any] = {
    "shape": [STATIC_DIM],
    "dtype": "float32",
    "feature_names": ["present", "gap_norm", "rel_left", "rel_same", "rel_right"],
    "units": "0/1; [0,1]（gap/scan_range）; one-hot",
    "semantics": (
        "自车当前车道走廊（中心线两侧 lane.width/2 + lat_margin + lane_span·lane.width，"
        "默认覆盖左右各 1 条相邻车道）内前方最近静态障碍：present=1 ⇔ 在 scan_range 内；"
        "gap_norm = clamp(gap,0,range)/range，gap = s_obj - s_ego - 0.5·L_obj"
        "（自车中心→障碍近面，与 env/expert/pure_pursuit_idm.py::_static_blocker_gap 同口径）；"
        "rel one-hot 顺序 = STATIC_REL_BUCKETS=(-1,0,+1)，+1 = 障碍在自车当前车道的左侧相邻车道"
        "（y 左正）；无发现 → 全 0"
    ),
    "scan_range_m": STATIC_SCAN_RANGE_M,
    "lat_margin_m": STATIC_LAT_MARGIN_M,
    "lane_span": STATIC_LANE_SPAN,
    "rel_buckets": list(STATIC_REL_BUCKETS),
    "source": "env.obs.static.scan_static_blocker（engine.get_objects() 的 BaseBuilding，仿 IDM）",
    "fallback": "metadrive 不可用 / ego 无 lane / 无建筑 → 全 0；旧数据（28 维 others）由 net 零填充 static 段",
}


#: 世界系地图状态布局（v3，A4 nav 修正）
WORLD_LAYOUT: dict[str, Any] = {
    "ego_world": {
        "shape": [1, EGO_WORLD_DIM],
        "dtype": "float32",
        "feature_names": ["x", "y", "theta"],
        "units": "m / m / rad（世界系，heading 为前进方向角）",
        "semantics": "t0 自车世界位姿；rollout/教师强制的位姿链锚点（不做 SE(2) 对齐、不进历史）",
    },
    "ego_world_mask": {"shape": [1], "dtype": "float32", "semantics": "1 = 自车存在"},
    "route_world": {
        "shape": [ROUTE_WORLD_MAX_POINTS, ROUTE_WORLD_DIM],
        "dtype": "float32",
        "feature_names": ["x", "y"],
        "units": "m（世界系）",
        "semantics": (
            "世界系路线折线：顶点 = 首段起点 + 每个路段的 road 中心终点（与 get_checkpoints 的 "
            "checkpoint 同口径）；覆盖整条路线（含已驶过部分）；不足 64 点用末点重复填充（mask=0）"
        ),
        "source": "navigation.checkpoints + map.road_network.graph（NodeNetworkNavigation）",
        "downstream": "net.mem.rebuild_nav_from_world：按新 ego 位姿重算 checkpoint/命令/route_completion",
    },
    "route_world_mask": {
        "shape": [ROUTE_WORLD_MAX_POINTS],
        "dtype": "float32",
        "semantics": "1 = 该折线点为真实顶点；0 = 补位（下游必须按 mask 过滤，补位点不得参与几何计算）",
    },
}


#: LD 采样布局（schema v6 规范）
LD_LAYOUT: dict[str, Any] = {
    "shape": [16, LD_DIM],
    "offsets_m": list(LD_OFFSETS_M),
    "slot_policy": (
        "当前车道（candidate priority 0）占满全部 offset（primary，5 槽）；其余候选车道按 "
        "offset 环优先填充（先 5 m 环所有车道，再 10/15/20/30 m），环内按车道优先级；"
        "共 16 槽，超预算的远端环被裁"
    ),
    "near_field": (
        "近场 5/10 m 采样点提供横向锚定与预瞄几何"
        "（v8 诊断臂 A1' 回退 v4 近场口径；lane 通道已删除）"
    ),
    "discard": "采样点超车道末端或投影失败 → 该槽 mask=0（通道自动降级）",
}


def schema_manifest(
    *,
    num_slots: int = 16,
    frames: int = 6,
    ld_dim: int = LD_DIM,
    od_dim: int = OD_DIM,
    others_dim: int = OTHERS_HEAD_DIM + len(road_class_labels()),
) -> dict[str, Any]:
    """返回 obs schema 清单（当前帧 + 6 帧历史 + 世界系状态 + 槽位策略 + others/static 布局）。"""
    return {
        "schema_version": 6,
        "frame": {
            "ego": {
                "shape": [1, EGO_DIM],
                "dtype": "float32",
                "feature_names": list(EGO_FEATURE_NAMES),
                "units": "v m/s; a m/s^2; yaw_rate rad/s; steer [-1,1]; curvature 1/m; prev_ds m; prev_dtheta rad",
                "semantics": "自车动力学 + 上一策略步动作（reserved 两维由调用方注入）",
            },
            "od": {
                "shape": [num_slots, od_dim],
                "dtype": "float32",
                "feature_names": list(OD_FEATURE_NAMES),
                "units": "dx/dy m（自车系 x 前/y 左）; vx/vy m/s（相对速度）; cos/sin 无量纲; L/W m; type_id 枚举",
                "semantics": "固定槽位 = track id（见 od_slot_policy）；presence=0 的槽保留最近盒内观测",
            },
            "od_mask": {
                "shape": [num_slots],
                "dtype": "float32",
                "semantics": "1 = 该槽本帧有效（已分配 track，od_id>=0）；与 od_presence 不同",
            },
            "od_id": {
                "shape": [num_slots],
                "dtype": "int64",
                "semantics": "episode 内稳定 track id；-1 = 空槽",
            },
            "od_presence": {
                "shape": [num_slots],
                "dtype": "float32",
                "semantics": "1 = 对象本帧在盒内被观测（特征新鲜）；0 = 出盒未释放（特征陈旧）/空槽",
            },
            "ld": {
                "shape": [num_slots, ld_dim],
                "dtype": "float32",
                "feature_names": list(LD_FEATURE_NAMES),
                "units": "dx/dy m; heading_rel rad; curvature 1/m（右转负）; speed_limit m/s（原始）; 线型 id 枚举",
                "semantics": (
                    "车道点通道（v6：offset {0,20,40,60,80} m；0 m 点 = ego 投影点提供近场锚，"
                    "当前车道 5 primary 槽 + 其余车道环填充，见 ld_layout）"
                ),
            },
            "ld_mask": {"shape": [num_slots], "dtype": "float32", "semantics": "1 = 该槽本帧有效"},
            "nav": {"shape": [1, NAV_FEATURE_DIM], "dtype": "float32", "feature_names": list(NAV_FEATURE_NAMES), "units": "m / one-hot / [0,1]", "semantics": "兼容保留；others 是规范输入"},
            "nav_mask": {"shape": [1], "dtype": "float32", "semantics": "1 = 导航可用"},
            "signal": {"shape": [1, SIGNAL_DIM], "dtype": "float32", "feature_names": list(SIGNAL_FEATURE_NAMES), "semantics": "恒 [0,0,0,1]（本项目无灯）"},
            "signal_mask": {"shape": [1], "dtype": "float32", "semantics": "恒 1"},
            "others": {"shape": [1, others_dim], "dtype": "float32", "semantics": "规范上下文输入（nav+speed_limit+signal+static+road_class，见 others_layout / static_layout）"},
            "others_mask": {"shape": [1], "dtype": "float32", "semantics": "1 = ego 存在（通道整体可用）"},
        },
        "history": {
            "window": f"{frames} 帧 @0.5 s（stride=5 env steps），旧→新，SE(2) 对齐到当前帧",
            "keys": {
                "ego_hist": {"shape": [frames, 1, EGO_DIM], "dtype": "float32"},
                "ego_hist_mask": {"shape": [frames, 1], "dtype": "float32"},
                "others_hist": {"shape": [frames, 1, others_dim], "dtype": "float32"},
                "others_hist_mask": {"shape": [frames, 1], "dtype": "float32"},
                "od_hist": {"shape": [frames, num_slots, od_dim], "dtype": "float32", "semantics": "同槽位跨帧身份一致（不做槽位重排）"},
                "od_hist_mask": {"shape": [frames, num_slots], "dtype": "float32"},
                "od_id_hist": {"shape": [frames, num_slots], "dtype": "int64", "semantics": "各帧 od_id（不做 SE(2) 变换）；缺帧 -1"},
                "od_presence_hist": {"shape": [frames, num_slots], "dtype": "float32", "semantics": "各帧 presence；缺帧 0"},
                "ld_hist": {"shape": [frames, num_slots, ld_dim], "dtype": "float32"},
                "ld_hist_mask": {"shape": [frames, num_slots], "dtype": "float32"},
                "hist_valid": {
                    "shape": [frames],
                    "dtype": "float32",
                    "semantics": "1 = 该历史 slot 来自真实帧（精确 (episode, step-stride*j) 查表命中）；0 = 预热补位/过滤洞",
                },
            },
            "storage": "per_frame_v2：npz 按帧存当前帧通道 + od_id_hist/od_presence_hist；其余历史由 (episode_id, step) 精确查表重建",
        },
        "od_slot_policy": OD_SLOT_POLICY,
        "others_layout": OTHERS_LAYOUT,
        "static_layout": STATIC_LAYOUT,
        "world_layout": WORLD_LAYOUT,
        "ld_layout": LD_LAYOUT,
    }
