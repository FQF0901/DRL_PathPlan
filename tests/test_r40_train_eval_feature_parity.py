"""R40-V5：train runner ↔ eval 想象链 **残差特征同源守卫**（小样本逐位；GPT §六.10 / 猜想 4）。

对拍主体（**同一批 transition、同一模型实例、同一残差头实例**；零训练、全 ``no_grad``）：

- train 侧 = ``tools/train_residual.py``（正式 runner；R41 第一批-B 链入库自包含：
  数据/obs = ``pipeline.residual_dataset``，特征/目标 = ``pipeline.residual_features``）：
  ``build_pairs``（row t → step t+5）→ ``build_supervision``（cache 关、现场重算）产出特征
  26 列（8+8+2+4+4）；
- eval 侧 = 同一模型实例的 net 内想象链（``set_world_state_api("analytic_learned",
  speed_alpha=1.0)`` + ``set_world_residual_scale(1.0)`` + ``model(obs, forward_mode=
  "world_model_teacher_forcing")``），head forward hook 捕获送入残差头的特征 / ΔA_θ /
  归一化输入 / ``apply_readout_residual`` 应用掩码（k=1 真实单步 transition）；
- 逐项门：5 特征键 + ΔA_θ + 归一化 + mask，**max_abs_diff = 0**（逐 chunk，与两侧 runner
  同 batching）。

口径出处 = round40/v1_feat_gate.py（全量 250,622 transition 硬门；本测试 = 同口径小样本
守卫，大资源缺失时 skip；无资源核心守卫在 ``tests/test_r41_residual_chain_selfcontained.py``）。
几何维度：GPU 张量按 batch 形状分块计算（TF32 GEMM tiling），**跨 batch 拼接不是同源口径**
⇒ 两侧同 chunk。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import tools.train_residual as runner
from net.world_residual import READOUT_RESIDUAL_INDICES, ReadoutResidualHead

#: 小样本规模（守卫用；两个 chunk 起 ⇒ 覆盖逐 chunk 对拍）
N_PAIRS = 128
CHUNK = 64
#: 特征键序（= runner.FEATURE_KEYS；train/eval 两侧同一契约）
FEATURE_KEYS = runner.FEATURE_KEYS


# --------------------------------------------------------------------------- #
# 装配（资源缺失 ⇒ skip；本机资源齐全时运行）
# --------------------------------------------------------------------------- #
def _missing_resources() -> list[str]:
    paths = {
        "runner_config": runner.DEFAULT_CONFIG,
        "data": runner.ROOT / "datasets/path_exec_v1/manifest.json",
        "base_dataset": runner.ROOT / "datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz",
        "init": runner.ROOT / "runs/p1_v2/round22/M0/ckpt_epoch002.pt",
        "model_config": runner.ROOT / "runs/p1_v2/round22/M0/model.snapshot.yaml",
        "ckpt": runner.ROOT / "runs/round36/j2/t1_keep_res/ckpt_epoch001.pt",
    }
    return [name for name, path in paths.items() if not path.exists()]


@pytest.fixture(scope="module")
def parity_setup():
    """train 侧：正式 runner build_pairs + build_supervision（cache 关）。"""
    missing = _missing_resources()
    if missing:
        pytest.skip(f"R40-V5 特征同源守卫资源缺失：{missing}（见 docstring 复现命令）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = runner.parse_args([
        "--max-pairs", str(N_PAIRS), "--chunk", str(CHUNK), "--seed", "0",
        "--cache", "", "--device", str(device),
    ])
    data = runner.build_pairs(args)
    supervision = runner.build_supervision(args, device, data)
    assert not supervision["cache_hit"], "守卫必须现场重算（禁缓存）"
    return args, device, data, supervision


def _tiny_perturbed_head(device: torch.device) -> ReadoutResidualHead:
    """非零残差头（末层零初始化 ⇒ 需扰动才能对拍非平凡 Δ；固定种子 ⇒ 可复现）。"""
    torch.manual_seed(1234)
    head = ReadoutResidualHead(hidden=64)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.1)
    return head.eval().to(device)


def _split_train_features(raw: np.ndarray) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for key, width in zip(FEATURE_KEYS, runner.FEATURE_WIDTHS):
        out[key] = torch.as_tensor(raw[:, offset : offset + width], dtype=torch.float32)
        offset += width
    assert offset == raw.shape[1]
    return out


@pytest.fixture(scope="module")
def eval_capture(parity_setup):
    """eval 侧：net 内想象链（analytic_learned）逐 chunk 捕获（取 k=1 首次调用）。"""
    args, device, data, supervision = parity_setup
    head = _tiny_perturbed_head(device)
    train_features = _split_train_features(np.asarray(supervision["features"]))

    all_head_records: list[dict] = []
    all_norm_records: list[torch.Tensor] = []
    all_apply_records: list[torch.Tensor] = []

    def _head_hook(module, hook_args, hook_kwargs, output):
        all_head_records.append({
            "features": {key: hook_kwargs[key].detach().to("cpu").clone() for key in hook_kwargs},
            "delta": output.detach().to("cpu").clone(),
        })

    def _norm_hook(module, hook_args):
        all_norm_records.append(hook_args[0].detach().to("cpu").clone())

    import net.world_residual as wr

    original_apply = wr.apply_readout_residual

    def _apply_hook(*, base_view, delta):
        payload = original_apply(base_view=base_view, delta=delta)
        all_apply_records.append(payload["mask"].detach().to("cpu").clone())
        return payload

    model = supervision["model"]
    model.set_world_state_api("analytic_learned", speed_alpha=1.0)
    model.set_world_residual_scale(1.0)
    model.set_world_residual_head(head)
    head_handle = head.register_forward_hook(_head_hook, with_kwargs=True)
    net_handle = head.net.register_forward_pre_hook(_norm_hook)
    wr.apply_readout_residual = _apply_hook
    try:
        from pipeline.residual_dataset import build_obs, load_base_dataset, resolve_path

        rows = data["rows"]
        main_ids = np.asarray(data["main_ids"], dtype=np.int64)
        base_dataset = load_base_dataset(resolve_path(Path(args.base_dataset)))
        with torch.no_grad():
            for start in range(0, int(main_ids.size), CHUNK):
                stop = min(start + CHUNK, int(main_ids.size))
                obs = build_obs(rows, main_ids[start:stop], device=device, base_dataset=base_dataset)
                before = (len(all_head_records), len(all_norm_records), len(all_apply_records))
                model(obs, forward_mode="world_model_teacher_forcing")
                assert len(all_head_records) > before[0], "eval 链未调用残差头（装配失配）"
                assert len(all_norm_records) > before[1] and len(all_apply_records) > before[2]
    finally:
        wr.apply_readout_residual = original_apply
        head_handle.remove()
        net_handle.remove()

    # 逐 chunk 取首次调用（= k=1 真实单步 transition；口径 = v1_feat_gate）
    n, chunk = int(main_ids.size), CHUNK
    n_chunks = len(range(0, n, chunk))
    # 每个前向内头被调用 steps 次（k=1..5）；按批切分取各批首次
    calls_per_forward = len(all_head_records) // n_chunks
    assert calls_per_forward >= 1 and len(all_head_records) == calls_per_forward * n_chunks
    first_head = [all_head_records[index * calls_per_forward] for index in range(n_chunks)]
    first_norm = [all_norm_records[index * calls_per_forward] for index in range(n_chunks)]
    first_apply = [all_apply_records[index * calls_per_forward] for index in range(n_chunks)]
    return {
        "device": device,
        "head": head,
        "train_features": train_features,
        "captured_features": [record["features"] for record in first_head],
        "captured_delta": [record["delta"] for record in first_head],
        "captured_norm": first_norm,
        "captured_apply_mask": first_apply,
        "n": n,
        "chunk": chunk,
        "calls_per_forward": calls_per_forward,
    }


# --------------------------------------------------------------------------- #
# 判定：逐 chunk 逐位（与两侧 runner 同 batching）
# --------------------------------------------------------------------------- #
def _assert_bitwise_chunks(name: str, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    assert pairs, f"{name}: 无对拍 chunk"
    for index, (a, b) in enumerate(pairs):
        fa = a.reshape(a.shape[0], -1).to(torch.float32).numpy()
        fb = b.reshape(b.shape[0], -1).to(torch.float32).numpy()
        assert fa.shape == fb.shape, f"{name} chunk{index}: 形状 {fa.shape} vs {fb.shape}"
        diff = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
        worst = float(diff.max()) if diff.size else 0.0
        assert np.array_equal(fa, fb), (
            f"{name} chunk{index}: 非逐位（max_abs_diff={worst:g}）"
        )


def test_captured_feature_keys_and_contract(eval_capture) -> None:
    """eval 捕获键集 = train 侧键集；列号契约 = (0,3,4,5)。"""
    for call in eval_capture["captured_features"]:
        assert sorted(call) == sorted(FEATURE_KEYS)
    assert sorted(eval_capture["train_features"]) == sorted(FEATURE_KEYS)
    assert tuple(READOUT_RESIDUAL_INDICES) == runner.TARGET_COLUMNS == (0, 3, 4, 5)
    assert eval_capture["calls_per_forward"] >= 1


def test_feature_items_bitwise_train_runner_vs_eval_chain(eval_capture) -> None:
    """5 特征键：train runner 输出 vs eval 想象链内部实调（逐 chunk 逐位）。"""
    train_features = eval_capture["train_features"]
    captured = eval_capture["captured_features"]
    n, chunk = eval_capture["n"], eval_capture["chunk"]
    for key in FEATURE_KEYS:
        pairs = []
        for index, start in enumerate(range(0, n, chunk)):
            stop = min(start + chunk, n)
            pairs.append((train_features[key][start:stop], captured[index][key]))
        _assert_bitwise_chunks(f"features.{key}", pairs)


def test_delta_output_bitwise_train_features_vs_eval_chain(eval_capture) -> None:
    """ΔA_θ：同一头对 train 特征求值 vs eval 链内实调输出（逐 chunk 逐位）。"""
    head = eval_capture["head"]
    train_features = eval_capture["train_features"]
    captured_delta = eval_capture["captured_delta"]
    n, chunk = eval_capture["n"], eval_capture["chunk"]
    pairs = []
    with torch.no_grad():
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            delta_train = head(**{
                key: train_features[key][start:stop].to(eval_capture["device"])
                for key in FEATURE_KEYS
            })
            pairs.append((delta_train.cpu(), captured_delta[len(pairs)]))
    _assert_bitwise_chunks("delta", pairs)


def test_normalization_bitwise_head_input(eval_capture) -> None:
    """归一化 26 维（head.net 输入）：eval 链内捕获 vs 同一头对 train 特征调用（逐 chunk 逐位）。"""
    head = eval_capture["head"]
    train_features = eval_capture["train_features"]
    n, chunk = eval_capture["n"], eval_capture["chunk"]
    norm_records: list[torch.Tensor] = []

    def _norm_hook(module, hook_args):
        norm_records.append(hook_args[0].detach().to("cpu").clone())

    handle = head.net.register_forward_pre_hook(_norm_hook)
    try:
        with torch.no_grad():
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                head(**{
                    key: train_features[key][start:stop].to(eval_capture["device"])
                    for key in FEATURE_KEYS
                })
    finally:
        handle.remove()
    assert len(norm_records) == len(eval_capture["captured_norm"])
    _assert_bitwise_chunks("normalization", list(zip(norm_records, eval_capture["captured_norm"])))


def test_apply_mask_bitwise_eval_chain_vs_base_present(eval_capture) -> None:
    """应用掩码：eval 链 ``apply_readout_residual`` 实捕 vs train 侧 ``base_present>0.5``（逐 chunk 逐位）。"""
    train_features = eval_capture["train_features"]
    captured_mask = eval_capture["captured_apply_mask"]
    n, chunk = eval_capture["n"], eval_capture["chunk"]
    pairs = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        base_present = (train_features["base_present"][start:stop] > 0.5).to(torch.float32)
        pairs.append((base_present, captured_mask[len(pairs)]))
    _assert_bitwise_chunks("apply_mask", pairs)


# --------------------------------------------------------------------------- #
# 轻量结构守卫（无资源依赖）：runner CLI / config / 参数门常量同源
# --------------------------------------------------------------------------- #
def test_runner_cli_declares_dry_run_and_check_features() -> None:
    args = runner.parse_args(["--dry-run"])
    assert args.dry_run is True and args.check_features is False
    args = runner.parse_args(["--check-features"])
    assert args.check_features is True
    # config 值 = round39/b2 默认口径（seed/split/batching/loss）
    assert (args.seed, args.chunk, args.batch_size, args.val_frac) == (0, 256, 2048, 0.1)
    assert args.loss == "smooth_l1" and args.hidden == 64
    assert runner.FEATURE_WIDTHS == (8, 8, 2, 4, 4)


def test_runner_cli_overrides_config_and_rejects_unknown_config_key(tmp_path) -> None:
    args = runner.parse_args(["--seed", "7", "--epochs", "3"])
    assert args.seed == 7 and args.epochs == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 0\nnope: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="未知键"):
        runner.parse_args(["--config", str(bad), "--dry-run"])


def test_param_cap_constants_are_unified_with_param_probe() -> None:
    """残差预算常量：net.world_residual.TOTAL_PARAM_CAP == net.param_probe.TOTAL_CAP == 2,000,000。"""
    import net.param_probe as param_probe
    import net.world_residual as world_residual

    assert world_residual.TOTAL_PARAM_CAP == param_probe.TOTAL_CAP == 2_000_000
    assert not hasattr(world_residual, "TOTAL_PARAM_LIMIT")  # R39「×1.02」乘性规则不得回潮
    head = ReadoutResidualHead(hidden=64)
    report = world_residual.check_param_budget(head, model_total=2_000_000)
    assert report["model_total_cap"] == 2_000_000 and report["total_cap"] == 2_000_000
    with pytest.raises(AssertionError):
        world_residual.check_param_budget(head, model_total=2_000_001)
