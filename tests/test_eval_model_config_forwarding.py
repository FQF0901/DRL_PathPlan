"""评测侧 model_config 透传守卫（2026-10-10 ``spatial`` 漏传事故）。

事故：``eval_runner`` 构造评测任务时只透传 hidden_dim/moe/policy/value/world_model/
plan_anchor，遗漏 ``spatial`` → stg3 排摸臂（``spatial.layers=3``）的 ckpt 在评测侧被按
默认 2 层建模型（``unexpected=10``）；fail-fast 将其拦为 rc=2（旧行为会静默错评）。

本测试钉住：``eval_runner.task_model_config`` 必须透传 ``spatial``，且构造出的模型确实
包含第 3 层 spatial 参数。
"""
from __future__ import annotations

from pipeline.eval_runner import task_model_config
from pipeline.stages import build_model


def test_spatial_forwarded_and_third_layer_present() -> None:
    cfg = {
        "hidden_dim": 128,
        "moe": {"experts": {"count": 8, "hidden_dim": 76}},
        "policy": {"trunk_hidden": 160, "attn_heads": 4, "attn_layers": 1},
        "value": {"net_hidden": 256},
        "world_model": {"rollout_steps": 6},
        "plan_anchor": {"enabled": False},
        "spatial": {"type": "message_passing", "layers": 3},
    }
    mc = task_model_config(cfg)
    assert mc["spatial"]["layers"] == 3
    model = build_model(mc)
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith("st_gnn.spatial.layers.2.") for n in names), "spatial 3 层未生效"
    assert mc["hidden_dim"] == 128
    assert mc["policy"]["trunk_hidden"] == 160
    assert mc["value"]["net_hidden"] == 256


def test_default_spatial_is_two_layers() -> None:
    mc = task_model_config({"hidden_dim": 128})
    model = build_model(mc)
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith("st_gnn.spatial.layers.1.") for n in names)
    assert not any(n.startswith("st_gnn.spatial.layers.2.") for n in names)
