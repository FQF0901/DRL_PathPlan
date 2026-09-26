"""``env.obs`` 观测通道包（schema v2）。

``obs_fingerprint``：观测实现（``env/obs/*.py`` 内容 + schema 版本）的短指纹。BC 专家数据集
在 meta 里记录采集时的指纹；训练侧加载时比对——**观测 scope/特征语义改动后必须重新采集
BC 数据**（2026-09-25 实测：OD/LD scope 从圆形 100 m 改为盒式后，旧数据训练出的策略在新
观测下行为完全不同）。

v2（2026-09-26）变更：OD 固定槽位（槽位 = track id，``od_id``/``od_presence`` 伴随数组）、
``others`` 规范上下文通道、6 帧 mem 历史含 ``od_id_hist``/``od_presence_hist``/``ego_hist``/
``others_hist``、scope 前 100 → 150 m。指纹带 ``v2-`` 前缀，旧数据集加载时必然不匹配。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["obs_fingerprint", "OBS_SCHEMA_VERSION"]

#: 观测 schema 版本（数据结构契约版本，独立于内容哈希）
OBS_SCHEMA_VERSION = 2


def obs_fingerprint() -> str:
    """返回 ``v2-<内容哈希 12 hex>``；目录缺失时返回空串。"""
    directory = Path(__file__).resolve().parent
    if not directory.is_dir():
        return ""
    digest = hashlib.md5()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"v{OBS_SCHEMA_VERSION}-{digest.hexdigest()[:12]}"
