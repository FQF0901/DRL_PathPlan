"""P2/P4 GAE λ 选项：CLI/config 覆盖路径与 0.95/0.98 档（预注册 §7.2）。

锁定：
- ``--lam``（CLI）优先于 config ``train.ppo.lam``；默认 0.95；非法 fail-fast；
- ``PPOConfig.lam`` 透传（trainer 的 GAE 调用消费该值，语义：λ 越大优势链越长）；
- 0.95 / 0.98 档在合成 buffer 上给出可区分的优势（同 γ 单变量）。
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from pipeline.buffer import RolloutBuffer
from pipeline.stages import _parse_args, _resolve_stage_c_lam
from pipeline.trainer import PPOConfig


def _channels() -> dict:
    return {
        "ego": (1, 8),
        "od": (2, 9),
        "ld": (2, 7),
        "nav": (1, 11),
        "signal": (1, 4),
    }


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = {}
    for name, shape in _channels().items():
        obs[name] = rng.normal(size=shape).astype(np.float32)
        obs[f"{name}_mask"] = np.ones(shape[:-1], dtype=np.float32)
    obs["hist_valid"] = np.ones(6, dtype=np.float32)
    return obs


def _add(buffer: RolloutBuffer, seed: int, *, reward: float, value: float = 0.0,
         terminated: bool = False) -> None:
    buffer.add_step(
        _frame(seed),
        pose=np.zeros(3, dtype=np.float32),
        action=np.array([1.0, 0.1], dtype=np.float32),
        logprob=-0.5,
        value=value,
        reward=reward,
        terminated=terminated,
        truncated=False,
    )


def _ns(value):
    return argparse.Namespace(lam=value)


def test_lam_cli_config_resolution_and_fail_fast() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.lam is None  # 未给 → 走 config/默认
    assert _resolve_stage_c_lam(args, {}) == pytest.approx(0.95)  # 现行默认
    assert _resolve_stage_c_lam(args, {"ppo": {"lam": 0.98}}) == pytest.approx(0.98)

    cli = _parse_args(["--stage", "C", "--lam", "0.98"])
    assert cli.lam == pytest.approx(0.98)
    # CLI 优先于 config
    assert _resolve_stage_c_lam(cli, {"ppo": {"lam": 0.9}}) == pytest.approx(0.98)

    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(SystemExit):
            _resolve_stage_c_lam(_ns(bad), {})
    with pytest.raises(SystemExit):
        _resolve_stage_c_lam(_ns("bad"), {})


def test_ppo_config_lam_passthrough() -> None:
    assert PPOConfig().lam == pytest.approx(0.95)
    assert PPOConfig(lam=0.98).lam == pytest.approx(0.98)


def test_lam_095_vs_098_gae_semantics() -> None:
    """同 γ=0.99 单变量：λ=0.98 的优势链更长（A_0 严格大于 λ=0.95）。"""
    def advantages(lam: float) -> np.ndarray:
        buffer = RolloutBuffer(4, channels=_channels())
        for index in range(4):
            _add(buffer, index, reward=1.0, terminated=index == 3)
        adv, _ = buffer.compute_gae(last_value=0.0, gamma=0.99, lam=lam)
        return adv

    adv_095 = advantages(0.95)
    adv_098 = advantages(0.98)
    gamma = 0.99
    manual_098 = sum((gamma * 0.98) ** k for k in range(4))
    assert adv_098[0] == pytest.approx(manual_098, rel=1e-6)
    assert adv_098[0] > adv_095[0] > 0.0
    np.testing.assert_allclose(adv_095[3], adv_098[3])  # 末帧 δ 不受 λ 影响
