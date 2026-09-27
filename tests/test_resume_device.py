"""跨设备 resume 回归（2026-09-27 事故：GPU 训练的 ckpt 在 CPU 载入优化器状态 → Adam cuda/cpu 混用崩溃）。

`load_optimizer_state` 必须在载入前把状态张量搬到**对应参数所在设备**。
"""
from __future__ import annotations

import pytest
import torch

from pipeline.trainer import load_optimizer_state, move_optimizer_state_to_device


def _make(device: str):
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features = torch.randn(8, 4, device=device)
    model(features).sum().backward()
    optimizer.step()
    return model, optimizer


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_state_moved_to_param_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("无 CUDA")
    _src_model, src_optimizer = _make("cpu")  # 模拟 "ckpt 在 CPU 上"
    state = src_optimizer.state_dict()
    _dst_model, dst_optimizer = _make(device)
    # 人为把目标优化器状态放到 CPU，模拟 torch.load(map_location="cpu")
    for entry in dst_optimizer.state.values():
        for key, value in list(entry.items()):
            if torch.is_tensor(value):
                entry[key] = value.to("cpu")
    assert load_optimizer_state(dst_optimizer, state) is True
    param_devices = {param.device.type for group in dst_optimizer.param_groups for param in group["params"]}
    for entry in dst_optimizer.state.values():
        for key, value in entry.items():
            if torch.is_tensor(value):
                assert value.device.type in param_devices, (key, value.device, param_devices)
    # 载入后能真正 step（此前会在这里抛 "Expected all tensors to be on the same device"）
    _dst_model(torch.randn(8, 4, device=device)).sum().backward()
    dst_optimizer.step()


def test_optimizer_state_none_and_incompatible() -> None:
    _model, optimizer = _make("cpu")
    assert load_optimizer_state(optimizer, None) is False
    assert load_optimizer_state(optimizer, {"state": {}, "param_groups": [{"params": [0], "lr": 1e-3}]}) is False


def test_move_optimizer_state_after_model_to_device() -> None:
    """事故回归（2026-09-27）：状态先在 CPU、模型随后搬到 device → 迁移后 step 不得抛设备错误。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _src_model, src_optimizer = _make("cpu")
    state = src_optimizer.state_dict()
    model, optimizer = _make(device)
    optimizer.load_state_dict(state)
    for entry in optimizer.state.values():  # 模拟 torch.load(map_location="cpu") 的结果
        for key, value in list(entry.items()):
            if torch.is_tensor(value):
                entry[key] = value.to("cpu")
    moved = move_optimizer_state_to_device(optimizer)
    if device == "cuda":
        assert moved > 0
    for entry in optimizer.state.values():
        for key, value in entry.items():
            if torch.is_tensor(value):
                assert value.device.type == device
    model(torch.randn(8, 4, device=device)).sum().backward()
    optimizer.step()  # 此前会在 Adam 内部抛 "found at least two devices"
