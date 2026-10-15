#!/usr/bin/env python3
"""R38-P2b：旧 ``PolicyHead`` 诊断**离线工具**（替代主 forward 内的 ``set_policy_diag``）。

背景（GPT §七-D 第 6 步）
-------------------------
R36 L-A 后 structured 模式旧头**默认不前向**；诊断经 ``set_policy_diag(True)`` 在**主 forward
内并行计算**旧头输出（``policy_action_mu`` / ``action_logstd``）。R38-P2b 收编后正式 forward
只保留一个动作出口 ⇒ 该诊断**离线化**：本工具在主 forward 之外重算同样两项（数值与在网口径
逐位一致：同一 ``encode()`` 令牌集合 + 同一 ``model.policy`` 实例前向）。

用法::

    # ① 在网模型（默认关路径）：--obs-pt 提供 torch.save 的 obs dict（CPU）
    tools/venv-python tools/diagnostics/legacy_policy_diag.py \\
        --ckpt runs/.../final.pt --config config/model.yaml --obs-pt /tmp/obs.pt --out /tmp/diag.json

    # ② legacy 独立复算（旧 ckpt 的 policy.* 键面；不载入完整模型）
    tools/venv-python tools/diagnostics/legacy_policy_diag.py \\
        --legacy-ckpt runs/.../final.pt --tokens-pt /tmp/tokens.pt --out /tmp/diag_legacy.json

产物：``--out`` JSON（张量以 sha256 + shape 摘要落盘；``--dump-pt`` 可另存原始张量）。
**只读**：不写任何 ckpt/模型，不改变正式 forward 数值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402


def _tensor_digest(tensor: Tensor) -> dict[str, Any]:
    flat = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(flat.shape)).encode("utf-8"))
    digest.update(str(flat.dtype).encode("utf-8"))
    digest.update(flat.numpy().tobytes())
    return {
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "sha256": digest.hexdigest(),
        "abs_max": float(flat.abs().max()) if flat.numel() else 0.0,
    }


def diag_policy_head(model: Any, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """离线旧头诊断：``encode(obs)`` 令牌集合 + ``model.policy`` 前向（不在 ``model.forward`` 内）。

    返回 ``{"policy_action_mu": (B,2), "action_logstd": (B,2), "action_mu_forward": (B,2)}``；
    ``action_mu_forward`` = 正式 forward 的动作出口（对照用；默认关路径与旧头同源）。
    """
    encoded = model.encode(obs)
    tokens = encoded["tokens"]
    key_mask = encoded["key_mask"]
    policy_action_mu, action_logstd = model.policy(tokens, key_mask)
    with torch.no_grad():
        forward_out = model(obs, rollout=True, world_model=True)
    return {
        "policy_action_mu": policy_action_mu.detach(),
        "action_logstd": action_logstd.detach(),
        "action_mu_forward": forward_out["action_mu"].detach(),
    }


def legacy_head_from_ckpt(ckpt: str, *, strict: bool = False) -> tuple[torch.nn.Module, dict[str, Any]]:
    """legacy 独立复算入口：旧 ckpt 的 ``policy.*`` 键 → :class:`tools.legacy.legacy_policy.PolicyHead`。

    只做键面形状适配（``policy.`` 前缀 + ``log_std→logstd`` 改名），返回 ``(head, summary)``；
    ``strict=True`` 时多余/缺失键 ⇒ ``RuntimeError``（默认宽松并逐键计数）。
    """
    from tools.legacy.legacy_key_mapper import LEGACY_META_KEYS
    from tools.legacy.legacy_policy import PolicyHead

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], Mapping):
                state = payload[key]
                break
    meta = dict(payload.get("meta") or {}) if isinstance(payload, Mapping) else {}
    policy_state = {
        key[len("policy."):]: value
        for key, value in dict(state).items()
        if str(key).startswith("policy.")
    }
    if "log_std" in policy_state:  # 契约 §4.4 改名族（旧 ckpt）
        policy_state["logstd"] = policy_state.pop("log_std")
    head = PolicyHead()
    missing, unexpected = head.load_state_dict(policy_state, strict=bool(strict))
    summary = {
        "ckpt": str(ckpt),
        "policy_keys_loaded": len(policy_state) - len(unexpected),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "legacy_meta": {key: meta.get(key) for key in LEGACY_META_KEYS if key in meta},
    }
    return head, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legacy_policy_diag.py")
    parser.add_argument("--ckpt", default=None, help="完整模型 ckpt（在网模型诊断）")
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--ld-mode", default="baseline")
    parser.add_argument("--obs-pt", default=None, help="torch.save 的 obs dict")
    parser.add_argument("--legacy-ckpt", default=None, help="仅旧 ckpt（独立 PolicyHead 复算）")
    parser.add_argument("--tokens-pt", default=None, help="--legacy-ckpt 时提供 {'tokens','key_mask'}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump-pt", default=None, help="可选：原始张量另存")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"tool": "tools/diagnostics/legacy_policy_diag.py", "read_only": True}
    if args.legacy_ckpt:
        if not args.tokens_pt:
            raise SystemExit("--legacy-ckpt 需要 --tokens-pt（{'tokens','key_mask'}）")
        head, summary = legacy_head_from_ckpt(args.legacy_ckpt)
        head.eval()
        payload = torch.load(args.tokens_pt, map_location="cpu", weights_only=False)
        with torch.no_grad():
            mu, logstd = head(payload["tokens"], payload["key_mask"])
        report["legacy_head"] = summary
        report["outputs"] = {"policy_action_mu": _tensor_digest(mu), "action_logstd": _tensor_digest(logstd)}
        raw = {"policy_action_mu": mu, "action_logstd": logstd}
    else:
        if not (args.ckpt and args.obs_pt):
            raise SystemExit("需要 --ckpt + --obs-pt（在网诊断）或 --legacy-ckpt + --tokens-pt（独立复算）")
        from pipeline import eval_runner as ev
        from pipeline.trainer import load_checkpoint

        config = ev.load_config(args.config)
        model = ev._build_model_from_config(ev.task_model_config(config))
        if args.ld_mode != "baseline":
            model.set_ld_mode(args.ld_mode)
        load_info = load_checkpoint(args.ckpt, model, strict=False)
        model.eval()
        obs = torch.load(args.obs_pt, map_location="cpu", weights_only=False)
        outputs = diag_policy_head(model, obs)
        report["ckpt"] = str(args.ckpt)
        report["load_info"] = {
            "missing_keys": len(load_info.get("missing_keys", [])),
            "unexpected_keys": len(load_info.get("unexpected_keys", [])),
        }
        report["outputs"] = {key: _tensor_digest(value) for key, value in outputs.items()}
        raw = outputs
    if args.dump_pt:
        torch.save(raw, args.dump_pt)
        report["dump_pt"] = str(args.dump_pt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
