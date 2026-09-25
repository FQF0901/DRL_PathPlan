"""``env.obs`` 观测通道包。

``obs_fingerprint``：观测实现（``env/obs/*.py`` 内容）的短指纹。BC 专家数据集在 meta 里
记录采集时的指纹；训练侧加载时比对——**观测 scope/特征语义改动后必须重新采集 BC 数据**
（2026-09-25 实测：OD/LD scope 从圆形 100 m 改为盒式 front100/rear50/left25/right25 后，
旧数据训练出的策略在新观测下行为完全不同）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["obs_fingerprint"]


def obs_fingerprint() -> str:
    """返回 ``env/obs/*.py`` 内容哈希（12 hex）。缺失目录时返回空串。"""
    directory = Path(__file__).resolve().parent
    if not directory.is_dir():
        return ""
    digest = hashlib.md5()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
