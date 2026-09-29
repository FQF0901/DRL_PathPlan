"""C1/G3 最小下一步：阶段 C 周期 ckpt（保存点，不引入 resume）。

锁定：
- ``stages.C.ckpt_every`` 解析（CLI ``--ckpt-every`` 优先；默认 25；0=关）；
- 保存点调度 = 每个 ``update % N == 0``（1 基），命名 ``<out>/ckpt_u<NNN>.pt``；
- 保存 payload 与 ``final.pt`` 同格式（``save_checkpoint`` 固定键集 + meta stage/update）。
"""

from __future__ import annotations

from pathlib import Path

import torch

from pipeline.stages import (
    _parse_args,
    _resolve_stage_c_ckpt_every,
    _stage_c_ckpt_path,
    _stage_c_periodic_updates,
)
from pipeline.trainer import save_checkpoint


def test_periodic_updates_schedule() -> None:
    assert _stage_c_periodic_updates(200, 25) == [25, 50, 75, 100, 125, 150, 175, 200]
    assert _stage_c_periodic_updates(2, 1) == [1, 2]
    assert _stage_c_periodic_updates(20, 0) == [], "0=关"
    assert _stage_c_periodic_updates(10, 100) == []
    assert _stage_c_periodic_updates(0, 25) == []


def test_ckpt_path_naming_is_zero_padded() -> None:
    assert _stage_c_ckpt_path(Path("runs/x"), 25) == Path("runs/x/ckpt_u025.pt")
    assert _stage_c_ckpt_path(Path("runs/x"), 200) == Path("runs/x/ckpt_u200.pt")


def test_resolve_stage_c_ckpt_every_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert _resolve_stage_c_ckpt_every(args, {}) == 25, "默认 25"
    assert _resolve_stage_c_ckpt_every(args, {"ckpt_every": 0}) == 0, "config 0=关"
    assert _resolve_stage_c_ckpt_every(args, {"ckpt_every": 50}) == 50

    args_cli = _parse_args(["--stage", "C", "--ckpt-every", "1"])
    assert _resolve_stage_c_ckpt_every(args_cli, {"ckpt_every": 50}) == 1, "CLI 优先"
    args_off = _parse_args(["--stage", "C", "--ckpt-every", "0"])
    assert _resolve_stage_c_ckpt_every(args_off, {"ckpt_every": 25}) == 0


def test_ckpt_payload_matches_final_format(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    path = _stage_c_ckpt_path(tmp_path, 25)
    saved = save_checkpoint(path, model, meta={"stage": "C", "update": 25})
    assert Path(saved) == path and path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {"model", "meta", "optimizer", "epoch", "val_metrics", "rng_state", "config_hash"}
    assert payload["meta"] == {"stage": "C", "update": 25}
