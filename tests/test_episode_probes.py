"""v4 ⑤：训练侧数据探针（``episodes/*``）——本 update 窗口内的 episode 统计。

锁定：

- ``episodes/termination_counts`` / ``episodes/mean_steps_by_reason`` /
  ``episodes/mean_return_by_reason``（+ ``episodes/count``）来自 collect 窗口内 done 的
  episode（跨 update 的回报累计，done 结算）；
- 终止原因与 eval_runner 同序：arrive_dest → collision → out_of_road → cut → max_step → other；
- 窗口在 update() 后清空（下一个 update 无新 episode → count=0）；
- 纯监控：不影响训练数值/参数。

不建 env、不 import metadrive（脚本化池 + stub 奖励 + stub 模型）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from pipeline.trainer import (
    PPOConfig,
    PPOTrainer,
    _SmokeModel,
    episode_termination_reason,
    episode_window_metrics,
)

OBS = {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


class _ScriptedPool:
    """按脚本返回记录（越过末尾则重复最后一条，保持 rollout 可继续）。"""

    num_envs = 1
    accepts_pre_step_labels = False

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = list(records)
        self._index = 0

    def reset(self) -> List[Dict[str, Any]]:
        return [self._next()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._next()]

    def _next(self) -> Dict[str, Any]:
        # 越过脚本末尾 → 非终局空步（保持 rollout 可继续；不产生 episode 记录）
        if self._index >= len(self._records):
            self._index += 1
            return {
                "obs": OBS,
                "info": {},
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "spec_id": 0,
            }
        record = self._records[self._index]
        self._index += 1
        return {
            "obs": OBS,
            "info": dict(record.get("info") or {}),
            "reward": float(record.get("reward", 0.0)),
            "terminated": bool(record.get("terminated", False)),
            "truncated": bool(record.get("truncated", False)),
            "spec_id": 0,
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def _make_trainer(pool: _ScriptedPool) -> PPOTrainer:
    torch.manual_seed(0)
    trainer = PPOTrainer(
        _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)}),
        pool,
        PPOConfig(epochs=1, minibatch_size=4, lr=0.0, device="cpu"),
        reward_adapter=_RewardStub(),
        probe_batch=None,
        logger=lambda _line: None,
    )
    trainer.adopt_obs([OBS])
    return trainer


def _collision_episode() -> List[Dict[str, Any]]:
    return [
        {"reward": 1.0},
        {"reward": 2.0},
        {"reward": 3.0, "terminated": True, "info": {"crash_vehicle": True}},
    ]


def test_episode_probe_counts_steps_return_by_reason() -> None:
    trainer = _make_trainer(_ScriptedPool(_collision_episode()))
    trainer.collect_rollout(3)
    metrics = trainer.update()
    episodes = metrics["episodes"]
    assert episodes["count"] == 1.0
    assert episodes["termination_counts"] == {"collision": 1.0}
    assert episodes["mean_steps_by_reason"] == {"collision": 3.0}
    assert episodes["mean_return_by_reason"] == {"collision": 6.0}, "回报 = 逐步奖励累计（1+2+3）"


def test_episode_window_resets_after_update() -> None:
    trainer = _make_trainer(_ScriptedPool(_collision_episode()))
    trainer.collect_rollout(3)
    trainer.update()
    trainer.collect_rollout(2)  # 无新 episode 结束
    second = trainer.update()
    assert second["episodes"]["count"] == 0.0
    assert second["episodes"]["termination_counts"] == {}
    assert second["episodes"]["mean_steps_by_reason"] == {}
    assert second["episodes"]["mean_return_by_reason"] == {}


def test_episode_return_accumulates_across_updates() -> None:
    """episode 跨 update 边界：回报累计不丢失（第 1 段 1.0 + 第 2 段 2.0 = 3.0）。"""
    records = [{"reward": 1.0}] + [
        {"reward": 2.0, "terminated": True, "info": {"arrive_dest": True}},
    ]
    trainer = _make_trainer(_ScriptedPool(records))
    trainer.collect_rollout(1)
    first = trainer.update()
    assert first["episodes"]["count"] == 0.0, "第 1 步未终局"
    trainer.collect_rollout(1)
    second = trainer.update()
    assert second["episodes"]["termination_counts"] == {"arrive_dest": 1.0}
    assert second["episodes"]["mean_return_by_reason"] == {"arrive_dest": 3.0}


def test_termination_reason_taxonomy_matches_eval_order() -> None:
    assert episode_termination_reason({"arrive_dest": True}, terminated=True) == "arrive_dest"
    assert episode_termination_reason({"crash": True}, terminated=True) == "collision"
    assert episode_termination_reason({"out_of_road": True}, terminated=True) == "out_of_road"
    assert episode_termination_reason({}, cut=True) == "cut"
    assert episode_termination_reason({}, truncated=True) == "max_step"
    assert episode_termination_reason({"max_step": True}, terminated=True) == "max_step"
    assert episode_termination_reason({}, terminated=True) == "other"
    # 优先级：arrive_dest 优先于同时出现的 crash 标志（与 eval_runner 同序）
    assert episode_termination_reason({"arrive_dest": True, "crash": True}, terminated=True) == "arrive_dest"


def test_episode_window_metrics_empty_and_multi_reason() -> None:
    assert episode_window_metrics([]) == {
        "count": 0.0,
        "termination_counts": {},
        "mean_steps_by_reason": {},
        "mean_return_by_reason": {},
    }
    summary = episode_window_metrics(
        [
            {"reason": "collision", "steps": 10, "return": -5.0},
            {"reason": "collision", "steps": 20, "return": -15.0},
            {"reason": "max_step", "steps": 600, "return": 8.0},
        ]
    )
    assert summary["termination_counts"] == {"collision": 2.0, "max_step": 1.0}
    assert summary["mean_steps_by_reason"] == {"collision": 15.0, "max_step": 600.0}
    assert summary["mean_return_by_reason"] == {"collision": -10.0, "max_step": 8.0}
