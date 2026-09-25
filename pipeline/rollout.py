"""B1 内部 rollout 包装（P2 契约 §2 / §4）：策略自回归 ×6 + world model 推演。

角色
----
``net/model.py`` 的 ``DrivingModel`` 负责网络内部实现；本模块是**推理/评估侧的
薄包装**，把外部 numpy 观测 ↔ 模型张量、batch 维、设备、``no_grad``、确定性采样
统一掉，并给出契约里约定的输出口径：

- ``action_mu (B,2)`` / ``action_logstd (B,2)`` / ``value (B,1)``；
- ``traj_xy (B,6,2)``：B1 自回归产生的 3 s 轨迹（0.5 s 一点，自车系）；
- ``od_pred (B,6,16,5)`` / ``ld_pred (B,6,16,4)``：世界模型推演；
- 若模型额外给出 ``action_seq (B,6,2)``（自回归的 6 个动作），可用
  :func:`reference_from_actions` 经 ``env/tracking.interpolate`` 展开成 MPC/LQR 用的
  30 点参考（单一真源）；否则 6 点 ``traj_xy`` 本身也能被 ``LqrTracker.set_reference``
  接受为 ``(N,3)`` 参考。

模型接口（鸭子类型，任一即可）
------------------------------
``model.rollout(obs) -> dict``（§2 的 B1 实现，内部做自回归 ×6 + WM，``no_grad`` 可用）
或 ``model.forward(obs) / model(obs) -> dict``（至少含 ``action_mu``）。

确定性
--------
``deterministic=True``（默认）直接取 ``action_mu`` 并裁剪到动作界；``False`` 时用
``torch.Generator(seed)``（无 torch 则 ``np.random.default_rng(seed)``）采样 logstd 的
高斯策略——同一 seed/权重/输入必然同一输出。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from pipeline.buffer import DEFAULT_CHANNELS as _DEFAULT_OBS_SHAPES

__all__ = ["RolloutRunner", "reference_from_actions", "tensorize_obs"]

#: 契约输出形状（``B`` 为 batch 维占位）；仅校验存在的键。
_EXPECTED_SHAPES: dict[str, tuple] = {
    "action_mu": ("B", 2),
    "action_logstd": ("B", 2),
    "value": ("B", 1),
    "traj_xy": ("B", 6, 2),
    "od_pred": ("B", 6, 16, 5),
    "ld_pred": ("B", 6, 16, 4),
    "router_logits": ("B", 8),
}


def reference_from_actions(actions: Any, *, dt: float = 0.5, hz: int = 10) -> np.ndarray:
    """动作序列 → 10 Hz 参考轨迹（``interpolate`` 单一真源）。

    形状口径（避免歧义）：``(N,2)`` = **一条** 动作序列（N 个策略步）；
    ``(B,N,2)`` = batch 的动作序列。返回 ``(N*dt*hz, 3)`` / ``(B, N*dt*hz, 3)``。
    单步动作也走同一路径（``(2,)`` → ``(dt*hz, 3)``）。
    """
    from env.tracking import interpolate

    arr = np.asarray(actions, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("actions 不能为空")
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"单步动作需要 (2,)，收到 shape={arr.shape}")
        return interpolate(arr, dt=dt, hz=hz)
    if arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"动作序列需要 (N,2)，收到 shape={arr.shape}")
        return interpolate(arr, dt=dt, hz=hz)
    if arr.ndim == 3:
        if arr.shape[2] != 2:
            raise ValueError(f"batch 动作序列需要 (B,N,2)，收到 shape={arr.shape}")
        return np.stack([interpolate(seq, dt=dt, hz=hz) for seq in arr], axis=0)
    raise ValueError(f"actions 需要 (2,)/(N,2)/(B,N,2)，收到 shape={arr.shape}")


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:  # noqa: BLE001 - 无 torch 时退化为 numpy 直通（测试/村环境可用）
        return None


def _ndim(value: Any) -> int:
    """张量/np 数组的维数（torch.Tensor 也有 .ndim）。"""
    ndim = getattr(value, "ndim", None)
    return int(ndim) if ndim is not None else int(np.ndim(value))


def _looks_batched(obs: Mapping[str, Any]) -> bool:
    """按已知单帧形状推断 batch 维（``ego (1,8)`` vs ``(B,1,8)``）。"""
    for name in sorted(obs):
        shape = _DEFAULT_OBS_SHAPES.get(name)
        if shape is None:
            continue
        ndim = _ndim(obs[name])
        if ndim == len(shape) + 1:
            return True
        if ndim == len(shape):
            return False
    # 兜底：无已知键时以 ndim>=3 视为带 batch 维
    return any(_ndim(value) >= 3 for value in obs.values())


def tensorize_obs(obs: Mapping[str, Any], *, device: Any = None, dtype: Any = None) -> dict:
    """numpy/tensor 混合的 obs → torch 张量（无 torch 时原样 numpy）。

    ``dtype`` 只作用于浮点张量；整数/布尔键保持原 dtype。
    """
    torch = _try_import_torch()
    if torch is None:
        return {key: np.asarray(value) for key, value in obs.items()}
    out: dict = {}
    for key, value in obs.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        out[key] = tensor
    return out


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


class RolloutRunner:
    """B1 推理包装：观测 → 动作 / 6 点轨迹 / OD·LD 推演（numpy in/out）。"""

    def __init__(
        self,
        model: Any,
        *,
        device: Any = None,
        deterministic: bool = True,
        horizon: int = 6,
        dtype: Any = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
        seed: int = 0,
        strict_shapes: bool = True,
        eval_mode: bool = True,
    ):
        """
        Args:
            model: 实现 ``rollout(obs)`` 或 ``forward(obs)`` 的模型（``nn.Module`` 或鸭子类型）。
            device: torch 设备（``None`` = 模型现有设备 / CPU）。
            deterministic: True 取 ``action_mu``；False 按 logstd 采样（固定 seed）。
            horizon: B1 步数（默认 6 = 3 s / 0.5 s）。
            dtype: 浮点张量 dtype（``None`` = 不转换）。
            action_low / action_high: 动作裁剪界（policy 头有界，仍做一次保护性 clip）。
            seed: 采样随机种子（仅 ``deterministic=False`` 时使用）。
            strict_shapes: 校验契约形状，形状不符直接报错（早期抓网络 bug）。
            eval_mode: 构造时调用 ``model.eval()``（若存在），关闭 dropout/BN 更新。
        """
        self.model = model
        self.deterministic = bool(deterministic)
        self.horizon = int(horizon)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.seed = int(seed)
        self.strict_shapes = bool(strict_shapes)
        self._torch = _try_import_torch()
        if device is None and self._torch is not None:
            device = getattr(model, "device", None)
            if device is None:
                try:
                    device = next(model.parameters()).device  # nn.Module 默认设备
                except Exception:  # noqa: BLE001 - 鸭子类型模型没有 parameters
                    device = None
        self.device = device
        self.dtype = dtype if dtype is not None else (getattr(self._torch, "float32", None) if self._torch else None)
        self._generator = None
        self.seed_generator()
        if eval_mode and hasattr(model, "eval"):
            model.eval()

    # ------------------------------------------------------------------ 主入口
    def infer(self, obs: Mapping[str, Any], *, batched: Optional[bool] = None) -> dict:
        """跑一次 B1（优先 ``model.rollout``），返回 numpy 输出 dict（形状见模块 docstring）。"""
        return self._run(obs, method="rollout", batched=batched)

    #: 语义别名：一次完整的 B1 计划
    plan = infer

    def act(self, obs: Mapping[str, Any], *, batched: Optional[bool] = None) -> np.ndarray:
        """闭环控制用的单步动作 ``(B,2)``（优先 ``model.forward``，避免多余的 B1 展开）。"""
        outputs = self._run(obs, method="forward", batched=batched)
        if "action_mu" not in outputs:
            raise KeyError("模型输出缺少 action_mu，无法给出闭环动作")
        return self._action_from(outputs)

    def traj6(self, obs: Mapping[str, Any], *, batched: Optional[bool] = None) -> np.ndarray:
        """6 点 / 0.5 s 间隔的 3 s 轨迹（自车系）``(B,6,2)``。"""
        outputs = self.infer(obs, batched=batched)
        return self._require(outputs, "traj_xy")

    # ------------------------------------------------------------------ 内部
    def _run(self, obs: Mapping[str, Any], *, method: str, batched: Optional[bool] = None) -> dict:
        if batched is None:
            batched = _looks_batched(obs)
        inputs = tensorize_obs(obs, device=self.device, dtype=self.dtype)
        if not batched:
            inputs = {
                key: (value.unsqueeze(0) if hasattr(value, "unsqueeze") else np.expand_dims(value, 0))
                for key, value in inputs.items()
            }

        fn = getattr(self.model, method, None)
        if fn is None:
            fn = getattr(self.model, "forward", None)
        if fn is None:
            fn = self.model if callable(self.model) else None
        if fn is None:
            raise AttributeError(
                f"模型既没有 {method}()/forward() 也不可调用；B1 rollout 需实现 {method}(obs) -> dict"
            )

        torch = self._torch
        if torch is not None and any(isinstance(value, torch.Tensor) for value in inputs.values()):
            with torch.no_grad():
                raw = fn(inputs)
        else:
            raw = fn(inputs)
        outputs = {key: _to_numpy(value) for key, value in dict(raw).items()}
        if self.strict_shapes:
            batch_size = self._batch_size(inputs)
            self._validate(outputs, batch_size)
        if not batched:
            outputs = {key: (value[0] if isinstance(value, np.ndarray) and value.ndim > 0 else value)
                       for key, value in outputs.items()}
        return outputs

    @staticmethod
    def _batch_size(inputs: Mapping[str, Any]) -> int:
        for value in inputs.values():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])
        raise ValueError("obs 为空，无法确定 batch size")

    def _validate(self, outputs: Mapping[str, Any], batch_size: int) -> None:
        for key, value in outputs.items():
            expected = _EXPECTED_SHAPES.get(key)
            if expected is None or not isinstance(value, np.ndarray):
                continue
            tail = tuple(expected[1:])
            if value.ndim < 1 or int(value.shape[0]) != batch_size or tuple(value.shape[1:]) != tail:
                raise ValueError(
                    f"模型输出 {key} 形状 {value.shape} 不符合契约 "
                    f"(B={batch_size},{','.join(str(v) for v in tail)})"
                )
        if "traj_xy" in outputs and int(outputs["traj_xy"].shape[1]) != self.horizon:
            raise ValueError(f"traj_xy 的 horizon={outputs['traj_xy'].shape[1]} != 配置 {self.horizon}")

    def seed_generator(self, seed: Optional[int] = None) -> None:
        """（重）设采样随机种子，保证 ``deterministic=False`` 时可复现。"""
        if seed is not None:
            self.seed = int(seed)
        if self._torch is not None:
            self._generator = self._torch.Generator(device="cpu")
            self._generator.manual_seed(self.seed)
        else:
            self._generator = np.random.default_rng(self.seed)

    def _action_from(self, outputs: Mapping[str, Any]) -> np.ndarray:
        mu = self._require(outputs, "action_mu").astype(np.float32)
        if self.deterministic:
            action = mu
        else:
            logstd = outputs.get("action_logstd")
            if logstd is None:
                action = mu
            else:
                std = np.exp(np.asarray(logstd, dtype=np.float32))
                if self._torch is not None:
                    noise = self._torch.randn(mu.shape, generator=self._generator, dtype=self._torch.float32).numpy()
                else:
                    noise = self._generator.standard_normal(mu.shape).astype(np.float32)
                action = mu + std * noise
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    @staticmethod
    def _require(outputs: Mapping[str, Any], key: str) -> np.ndarray:
        if key not in outputs:
            raise KeyError(f"模型输出缺少 {key}；实际键={sorted(outputs)}")
        return np.asarray(outputs[key])
