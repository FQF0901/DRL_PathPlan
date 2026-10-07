"""obs schema v2 单测：OD 固定槽位（分配/驱逐/释放/presence）、6 帧 mem 顺序与 hist_valid、
id 跨帧一致性、schema 键与 dtype。

纯 NumPy + 假 env：用 ``FakeVehicle`` 顶替 ``env.obs.od.BaseVehicle`` 的 isinstance 检查，
不建 MetaDrive env（scope/几何语义在 meta drive smoke 里另有验证）。
"""

from __future__ import annotations

import numpy as np
import pytest

import env.obs.od as od_module
from env.obs.builder import ObservationBuilder
from env.obs.od import OD_ID_EMPTY, ODChannel
from env.obs.schema import schema_manifest


# --------------------------------------------------------------------------- #
# 假 env / 假车辆
# --------------------------------------------------------------------------- #

class FakeVehicle:
    """最小车辆接口：OD 通道只需要 id/position/velocity/heading/L/W。"""

    def __init__(self, name: str, x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0,
                 heading: float = 0.0):
        self.id = str(name)
        self.name = str(name)
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)
        self.heading_theta = float(heading)
        self.LENGTH = 4.5
        self.WIDTH = 1.8


class FakeEgo(FakeVehicle):
    def __init__(self, x: float = 0.0, y: float = 0.0, vx: float = 0.0, heading: float = 0.0):
        super().__init__("ego", x, y, vx, 0.0, heading)
        self.speed = float(abs(vx))
        self.last_speed = float(abs(vx))
        self.steering = 0.0
        self.last_heading_dir = np.array([np.cos(heading), np.sin(heading), 0.0])
        self.lane = None
        self.navigation = None

    def convert_to_local_coordinates(self, point, origin):
        """世界点 -> 自车系（x 前 / y 左）。"""
        p = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
        c, s = np.cos(self.heading_theta), np.sin(self.heading_theta)
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], 0.0], dtype=np.float64)


class FakeEngine:
    def __init__(self, objects):
        self.objects = {obj.id: obj for obj in objects}

    def get_objects(self):
        return self.objects


class FakeEnv:
    def __init__(self, ego: FakeEgo, others=(), step: int = 0):
        self._objects = [ego, *others]
        self.engine = FakeEngine(self._objects)
        self.episode_step = int(step)
        self.config = {}
        self.prev_policy_action = None

    @property
    def agent(self):
        return self._objects[0]

    def add(self, obj: FakeVehicle) -> None:
        self._objects.append(obj)
        self.engine.objects[obj.id] = obj


@pytest.fixture(autouse=True)
def _patch_base_vehicle(monkeypatch):
    monkeypatch.setattr(od_module, "BaseVehicle", FakeVehicle)


# --------------------------------------------------------------------------- #
# OD 固定槽位
# --------------------------------------------------------------------------- #

def test_slot_allocation_is_urgency_ordered_and_stable_across_frames():
    """新对象按紧迫度占第一个空槽；后续帧按 id 找槽（不重排）。"""
    ego = FakeEgo()
    near = FakeVehicle("near", 20.0, 0.0, vx=-2.0)  # ttc=10 -> cap 5
    far = FakeVehicle("far", 60.0, 0.0)  # ttc=inf -> cap 5，距离更远
    env = FakeEnv(ego, [near, far], step=0)
    channel = ODChannel()

    channel.build(env, None)
    ids = channel.companions()["od_id"]
    # near 与 far 同紧迫度（cap=5），距离近者先注册 -> track id 1 在 slot 0
    assert ids[0] == 1 and ids[1] == 2

    # 下一帧：near 远离出盒、far 变紧迫 —— 槽位不得交换
    near.position = np.array([400.0, 0.0, 0.0])
    far.position = np.array([8.0, 0.0, 0.0])
    far.velocity = np.array([-4.0, 0.0, 0.0])
    env.episode_step = 5
    feats, mask = channel.build(env, None)
    companions = channel.companions()
    assert companions["od_id"][0] == 1  # near 仍在 slot 0（身份稳定）
    assert companions["od_presence"][0] == 0.0  # 出盒
    assert companions["od_id"][1] == 2 and companions["od_presence"][1] == 1.0
    # mask 语义 = 槽位本帧有效（出盒未释放仍为 1）；presence 才表示本帧观测
    assert mask[0] == 1.0 and mask[1] == 1.0
    # 出盒槽保留最近一次盒内观测（stale），不是清零
    assert feats[0, 0] == pytest.approx(20.0)


def test_out_of_box_release_after_grace_period():
    """出盒 > 1.0 s（step 时间轴）→ 释放槽位；恰好 1.0 s 仍保留。"""
    ego = FakeEgo()
    target = FakeVehicle("t", 30.0, 0.0)
    env = FakeEnv(ego, [target], step=0)
    channel = ODChannel()
    channel.build(env, None)

    target.position = np.array([500.0, 0.0, 0.0])
    env.episode_step = 10  # 10 × 0.1 s = 1.0 s：不释放
    _, mask = channel.build(env, None)
    assert channel.companions()["od_id"][0] != OD_ID_EMPTY
    assert channel.companions()["od_presence"][0] == 0.0
    assert mask[0] == 1.0

    env.episode_step = 15  # 1.5 s > 1.0 s：释放
    _, mask = channel.build(env, None)
    assert channel.companions()["od_id"][0] == OD_ID_EMPTY
    assert channel.companions()["od_presence"][0] == 0.0
    assert mask[0] == 0.0

    target.position = np.array([25.0, 0.0, 0.0])
    env.episode_step = 20
    channel.build(env, None)
    assert channel.companions()["od_id"][0] != OD_ID_EMPTY  # 重新出现 -> 重新分配


def test_eviction_picks_longest_absent_out_of_box_slot():
    """槽满：紧迫度最高的新对象驱逐"最长未出现且已出盒"的槽；无可驱逐槽则丢弃。"""
    ego = FakeEgo()
    a = FakeVehicle("a", 10.0, 0.0)
    b = FakeVehicle("b", 20.0, 0.0)
    env = FakeEnv(ego, [a, b], step=0)
    channel = ODChannel(num_slots=2)
    channel.build(env, None)
    ids = channel.companions()["od_id"]
    assert set(ids.tolist()) == {1, 2}
    slot_a = int(np.where(ids == 1)[0][0])
    slot_b = int(np.where(ids == 2)[0][0])

    # b 出盒（未观测），a 在盒内；新对象 c 出现 -> 只能驱逐 b 的槽
    b.position = np.array([500.0, 0.0, 0.0])
    env.episode_step = 5
    channel.build(env, None)
    c = FakeVehicle("c", 4.0, 0.0, vx=-8.0)
    env.add(c)
    env.episode_step = 10
    channel.build(env, None)
    companions = channel.companions()
    assert companions["od_id"][slot_a] == 1
    assert companions["od_id"][slot_b] == 3  # c 接管 b 的槽
    assert companions["od_presence"][slot_b] == 1.0

    # 两槽都在盒内时：新对象被丢弃（不驱逐在盒对象）
    channel2 = ODChannel(num_slots=1)
    env2 = FakeEnv(FakeEgo(), [FakeVehicle("x", 12.0, 0.0)], step=0)
    channel2.build(env2, None)
    env2.add(FakeVehicle("y", 3.0, 0.0, vx=-10.0))
    env2.episode_step = 5
    channel2.build(env2, None)
    assert channel2.companions()["od_id"][0] == 1  # 仍是 x


def test_scope_front_150_and_rear_50():
    """v2 scope：前 150 / 后 50 / 左右 25（旧的 front=100 之外的 120 m 目标现在可见）。"""
    ego = FakeEgo()
    inside = FakeVehicle("inside", 120.0, 0.0)
    behind = FakeVehicle("behind", -60.0, 0.0)
    env = FakeEnv(ego, [inside, behind], step=0)
    channel = ODChannel()
    _, mask = channel.build(env, None)
    assert mask.sum() == 1.0  # inside 在盒内；behind 超出后 50 m
    assert channel.companions()["od_id"][0] == 1

    channel100 = ODChannel(front_m=100.0)
    _, mask100 = channel100.build(FakeEnv(FakeEgo(), [FakeVehicle("inside", 120.0, 0.0)]), None)
    assert mask100.sum() == 0.0


def test_reset_clears_slot_table():
    ego = FakeEgo()
    env = FakeEnv(ego, [FakeVehicle("a", 10.0, 0.0)], step=0)
    channel = ODChannel()
    channel.build(env, None)
    assert channel.companions()["od_id"][0] == 1
    channel.reset()
    assert channel.companions()["od_id"][0] == OD_ID_EMPTY


# --------------------------------------------------------------------------- #
# 6 帧 mem：顺序 / hist_valid / id 跨帧一致性
# --------------------------------------------------------------------------- #

def _make_builder() -> ObservationBuilder:
    # 只挂 ego/od/others：ld 需要真实车道几何，不在本文件构造
    return ObservationBuilder({"channels": ["ego", "od", "others"]})


def test_mem_history_order_valid_and_companions():
    """6 帧历史旧→新、hist_valid 由真实缓冲长度决定、od_id_hist 与当前槽位一致。"""
    builder = _make_builder()
    ego = FakeEgo()
    obj = FakeVehicle("obj", 30.0, 0.0, vx=-1.0)
    env = FakeEnv(ego, [obj], step=0)

    outputs = {}
    for step in (0, 5, 10):
        env.episode_step = step
        obj.position = np.array([30.0 - 0.5 * step, 0.0, 0.0])
        outputs[step] = builder.build(env, None)

    hist0 = outputs[0]
    assert hist0["hist_valid"].tolist() == [0, 0, 0, 0, 0, 1]
    assert hist0["od_id_hist"].dtype == np.int64
    assert hist0["od_id_hist"][5, 0] == hist0["od_id"][0]
    # 补位帧（valid=0）的 id/presence 必须是缺省值（与精确查表一致，不得伪造身份）
    assert hist0["od_id_hist"][:5, 0].tolist() == [-1] * 5
    assert hist0["od_presence_hist"][:5, 0].tolist() == [0.0] * 5

    hist5 = outputs[5]
    assert hist5["hist_valid"].tolist() == [0, 0, 0, 0, 1, 1]
    # 同一对象跨帧同槽位同 id
    assert hist5["od_id_hist"][4, 0] == hist5["od_id_hist"][5, 0] == hist5["od_id"][0]
    assert hist5["od_presence_hist"][4, 0] == 1.0

    hist10 = outputs[10]
    assert hist10["hist_valid"].tolist() == [0, 0, 0, 1, 1, 1]
    # 新→旧对齐：最新 slot 即当前帧 ego/od
    np.testing.assert_allclose(hist10["ego_hist"][5, 0], outputs[10]["ego"][0])
    np.testing.assert_allclose(hist10["od_hist"][5, 0], outputs[10]["od"][0])
    # 历史帧保留原始自车系特征（对齐到当前帧；本假 env 的 ego 静止，纯平移）
    assert hist10["od_hist"][4, 0, 0] == pytest.approx(hist10["od_hist"][5, 0, 0] + 2.5, abs=1e-4)


def test_builder_output_keys_and_dtypes_match_schema_manifest():
    """builder 输出必须覆盖 schema_manifest 的键，且 od_id 家族是 int64。"""
    builder = _make_builder()
    env = FakeEnv(FakeEgo(), [FakeVehicle("a", 10.0, 0.0)], step=0)
    obs = builder.build(env, None)
    manifest = schema_manifest()

    required = set(manifest["frame"]) | set(manifest["history"]["keys"])
    missing = {key for key in required if key not in obs}
    # ld / nav / signal 未挂载到 builder 配置；其余（含 v2/v3/v4/v6 全部新键）必须存在
    assert not (
        missing
        - {
            "ld", "ld_mask", "ld_hist", "ld_hist_mask",
            "nav", "nav_mask", "signal", "signal_mask",
        }
    )

    assert obs["od_id"].dtype == np.int64
    assert obs["od_id_hist"].dtype == np.int64
    assert obs["od_presence"].dtype == np.float32
    assert obs["others"].shape == (1, 33)  # 11 + 1 + 4 + 5 + 12（v4 static 段）
    assert obs["others_hist"].shape == (6, 1, 33)
    assert obs["hist_valid"].shape == (6, )


def test_others_channel_layout_and_fallbacks():
    """others = nav(11) + speed_limit(1) + signal(4) + static(5) + road_class one-hot(12)（假 env 全回退）。"""
    from env.obs.others import OTHERS_HEAD_DIM, STATIC_OFFSET, road_class_labels
    from env.obs.static import STATIC_DIM

    env = FakeEnv(FakeEgo(), [], step=0)
    builder = _make_builder()
    obs = builder.build(env, None)
    assert obs["others"].shape == (1, OTHERS_HEAD_DIM + len(road_class_labels()))
    assert obs["others"].shape == (1, 33)
    assert obs["others_mask"][0] == 1.0  # ego 存在即整体可用
    np.testing.assert_allclose(obs["others"][0, 11], 0.0)  # 限速未知 -> 0
    np.testing.assert_allclose(obs["others"][0, 12:16], [0, 0, 0, 1])  # 无灯占位
    # v4 static 段：假 env 无 lane/建筑 → 全 0（present=0）
    assert STATIC_OFFSET == 16 and STATIC_DIM == 5
    np.testing.assert_allclose(obs["others"][0, STATIC_OFFSET:OTHERS_HEAD_DIM], 0.0)
    # road_class 段：map_info 不可用 → 全 0
    np.testing.assert_allclose(obs["others"][0, OTHERS_HEAD_DIM:], 0.0)
