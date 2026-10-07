"""``env.obs`` 观测通道包（schema v6）。

``obs_fingerprint``：观测实现（``env/obs/*.py`` 内容 + schema 版本）的短指纹。BC 专家数据集
在 meta 里记录采集时的指纹；训练侧加载时比对——**观测 scope/特征语义改动后必须重新采集
BC 数据**（2026-09-25 实测：OD/LD scope 从圆形 100 m 改为盒式后，旧数据训练出的策略在新
观测下行为完全不同）。

v2（2026-09-26）变更：OD 固定槽位（槽位 = track id，``od_id``/``od_presence`` 伴随数组）、
``others`` 规范上下文通道、6 帧 mem 历史含 ``od_id_hist``/``od_presence_hist``/``ego_hist``/
``others_hist``、scope 前 100 → 150 m。指纹带 ``v2-`` 前缀，旧数据集加载时必然不匹配。

v3（2026-10-01，A4 nav 修正）变更：新增世界系键 ``ego_world (1,3)`` + ``route_world (64,2)``
（+ 逐点 ``route_world_mask``），供 rollout / 教师强制按新位姿重算 nav；指纹前缀升为 ``v3-``。

v4（2026-10-02，P1-A 静态障碍可观测性）变更：``others`` 新增 static 段（静态障碍/岗亭
走廊扫描：present + gap_norm + 相对车道 one-hot，``others`` 28 → 33 维），供策略在
tollgate 上游 25–39 m 观测到被占车道；指纹前缀升为 ``v4-``（旧数据由 net 零填充 static 段）。

v5（2026-10-05，结构迭代 A）变更：LD offset 改远场 ``{20,40,60,80} m``（当前车道 4 primary
槽 + 其余车道环填充，总槽位 16 不变）；新增 ``lane (1,17)`` 当前车道块与 ``ttc (1,12)``
OD 槽位 TTC 上下文 token；指纹前缀升为 ``v5-``。

v6（2026-10-07，v8 结构重构）变更：删除 ``lane``/``ttc`` 通道（v5 的上下文 token 在 v8
不再使用）；LD offset 改为 ``{0,20,40,60,80} m``——0 m 点 = ego 投影点（近场横向锚定，
补偿 lane 删除），远场保留 20..80 m 前视；指纹前缀升为 ``v6-``（旧数据由 net 缺键回退，
新数据必须重新采集）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["obs_fingerprint", "OBS_SCHEMA_VERSION"]

#: 观测 schema 版本（数据结构契约版本，独立于内容哈希）
OBS_SCHEMA_VERSION = 6


def obs_fingerprint() -> str:
    """返回 ``v6-<内容哈希 12 hex>``；目录缺失时返回空串。"""
    directory = Path(__file__).resolve().parent
    if not directory.is_dir():
        return ""
    digest = hashlib.md5()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"v{OBS_SCHEMA_VERSION}-{digest.hexdigest()[:12]}"
