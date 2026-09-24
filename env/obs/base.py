"""观测通道基类 + 历史帧 SE(2) 对齐工具（L2）。

接口契约（跨线一致）::

    class ObservationChannel(ABC):
        name: str
        def build(self, env, spec) -> tuple[np.ndarray, np.ndarray]: ...   # (features, mask)

约定：
- ``features`` 形状 ``(N, F)`` float32，``N`` 是槽位数（定长通道 N=1），``F`` 是该通道特征维数；
- ``mask`` 形状 ``(N,)`` float32：1=该槽有效，0=填充 / 不可用（下游必须按 mask 屏蔽）。

自车系约定（已核对 0.4.3 源码后确认）：**x 前向、y 左向**，即
``p_world = p_ego_world + R(θ) @ p_ego``，``p_ego = R(-θ) @ (p_world - p_ego_world)``。

证据（不要被上游注释误导）：
- ``metadrive/base_class/base_object.py:1024-1029``：BaseVehicle 覆写
  ``convert_to_local_coordinates`` 为 ``[ret[1], -ret[0]]``；
- ``metadrive/tests/test_component/test_vehicle_coordinates.py:76-86``：heading=π/2 时
  断言 local ``[L/2, +W/2]`` 映射到世界 ``(x - w/2, y + l/2)``——面朝 +y 时 -x 是左侧，
  所以第二个分量是"左"；
- ``metadrive/component/navigation_module/node_network_navigation.py:301`` 的注释
  "+y is the right hand side" 与实现不符（上游注释错误，仅影响其变量命名）。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 仅类型标注；spec.py 属于 L1a，未就绪时也不影响本模块导入
    from env.scenario.spec import ScenarioSpec


def wrap_to_pi(x: float | np.ndarray) -> float | np.ndarray:
    """把角度（或角度数组）归一化到 (-π, π]。"""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class FrameAlignment:
    """历史帧 SE(2) 对齐规则：声明特征里哪些分量是"点 / 向量 / 单标量角度"。

    为什么需要显式声明：不同通道的特征语义不同，只有点需要平移补偿，
    向量只旋转，角度标量只减 Δθ；其余分量（长度/线型/限速等）在坐标变换下不变。
    """

    #: 位置分量对 (x_idx, y_idx)：按 -Δθ 旋转并补偿 ego 位移
    point_pairs: tuple[tuple[int, int], ...] = ()
    #: 向量分量对 (x_idx, y_idx)：只按 -Δθ 旋转（速度/朝向单位向量等）
    vector_pairs: tuple[tuple[int, int], ...] = ()
    #: 单标量角度维索引：φ_new = φ_old - Δθ
    angle_dims: tuple[int, ...] = ()


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def se2_align(
    features: np.ndarray,
    *,
    alignment: FrameAlignment = FrameAlignment(),
    delta_theta: float,
    delta_xy: np.ndarray,
    current_theta: float,
) -> np.ndarray:
    """把"旧自车系"下的特征重表达到"当前自车系"。

    推导（x 前向 / y 左向，右手系）::

        p_world = p_old_world + R(θ_old) p_old
        p_new   = R(-θ_new) (p_world - p_new_world)
                = R(-Δθ) p_old - R(-θ_new) Δp
        Δp = p_new_world - p_old_world,  Δθ = θ_new - θ_old

    点走上式；向量（无原点）只做 ``R(-Δθ)``；标量角度减 ``Δθ``。
    """
    out = np.array(features, dtype=np.float32, copy=True)
    if out.ndim != 2:
        raise ValueError(f"features 必须是 (N, F)，收到 shape={out.shape}")
    if out.size == 0 or not (alignment.point_pairs or alignment.vector_pairs or alignment.angle_dims):
        return out

    r_vec = _rot(-float(delta_theta))  # 向量/点方向旋转
    shift = _rot(-float(current_theta)) @ np.asarray(delta_xy, dtype=np.float32)

    for ix, iy in alignment.point_pairs:
        xy = out[:, [ix, iy]] @ r_vec.T - shift
        out[:, ix] = xy[:, 0]
        out[:, iy] = xy[:, 1]
    for ix, iy in alignment.vector_pairs:
        out[:, [ix, iy]] = out[:, [ix, iy]] @ r_vec.T
    for ix in alignment.angle_dims:
        out[:, ix] = wrap_to_pi(out[:, ix] - float(delta_theta))
    return out


def make_empty(num_slots: int, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """返回全零 (features, mask)，供通道在无数据时直接使用（mask 全 0）。"""
    return (
        np.zeros((int(num_slots), int(feature_dim)), dtype=np.float32),
        np.zeros((int(num_slots), ), dtype=np.float32),
    )


def safe_ego(env):
    """安全获取 ego 车辆对象；拿不到时返回 None。

    为什么需要：``BaseEnv.agent`` 在"尚未 reset / agent 非激活"时会 ``assert`` 失败
    （envs/base_env.py:745-752），而通道契约要求任何情况下都能返回（无效则 mask=0）。
    """
    try:
        return env.agent
    except Exception:  # noqa: BLE001
        return None


class ObservationChannel(ABC):
    """观测通道基类。

    子类只需实现 :meth:`build`；``name``/``feature_dim``/``alignment`` 为类属性，
    ``reset()`` 用于清空跨 step 缓存（默认无状态）。
    """

    name: str = "channel"
    feature_dim: int = 0
    #: 历史帧对齐规则（见 :class:`FrameAlignment`），由 FrameMemory 读取
    alignment: FrameAlignment = FrameAlignment()

    @abstractmethod
    def build(self, env, spec: "ScenarioSpec | None") -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(features (N,F) float32, mask (N,) float32)``；不得抛异常（不可用时返回全零+mask 0）。"""
        raise NotImplementedError

    def reset(self) -> None:
        """episode 开始（env.reset 后首次 build）时调用；默认无状态。"""
        return None
