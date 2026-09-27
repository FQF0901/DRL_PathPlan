#!/usr/bin/env python3
"""数据集 → 聚类硬标签 sidecar（``cluster_v<k>_assignments.npz``）批注工具。

lane B ①：Stage B 训练**只读** sidecar（禁止在线重算）；本工具在"生数据集时算一次"。
sidecar 内容：``cluster``(N,) int16（软目标 argmax）、``top1_margin``(N,) float32、
``rows`` / ``cluster_k`` / ``cluster_version`` / ``spec_hash`` / ``raw_dim`` / ``keep_dims_hash`` /
``obs_fingerprint``（数据集 meta）——**不落地 (N,8) 软分布**。

用法::

    tools/venv-python tools/annotate_clusters.py --dataset datasets/BTC<ts>_expert5k \
        --cluster-config config/clusters/default.yaml

``tools/collect_expert.py`` 收尾在有可用 spec 时会自动调用同一函数（见
``pipeline.clusters.annotate_assignments``）；本工具用于补标/重标已有数据集。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CLUSTER_CONFIG = "config/clusters/default.yaml"


def annotate(dataset: str, cluster_config: str = DEFAULT_CLUSTER_CONFIG, *, chunk_size: int = 2048) -> Path:
    """加载数据集 + spec → 写 sidecar；返回 sidecar 路径（供 collect_expert 复用）。"""
    from pipeline.clusters import annotate_assignments, load as load_clusters
    from pipeline.trainer import BCDataset

    bc = BCDataset.load(dataset)
    spec = load_clusters(cluster_config)
    return annotate_assignments(
        dataset_dir=dataset if os.path.isdir(dataset) else os.path.dirname(os.path.abspath(dataset)),
        count=int(bc.count),
        obs_batch_fn=bc.build_obs_batch,
        spec=spec,
        dataset_meta=getattr(bc, "meta", None),
        chunk_size=int(chunk_size),
    )


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="数据集 → 聚类硬标签 sidecar（cluster_v<k>_assignments.npz）")
    parser.add_argument("--dataset", required=True, help="数据集目录或 expert_bc.npz 路径")
    parser.add_argument("--cluster-config", default=DEFAULT_CLUSTER_CONFIG,
                        help=f"冻结聚类 spec（默认 {DEFAULT_CLUSTER_CONFIG}）")
    parser.add_argument("--chunk-size", type=int, default=2048, help="批大小（默认 2048）")
    args = parser.parse_args(argv)

    if not os.path.exists(args.dataset):
        print(f"[annotate_clusters] 数据集不存在：{args.dataset}", file=sys.stderr)
        return 2
    if not Path(args.cluster_config).is_file():
        print(f"[annotate_clusters] 聚类 spec 不存在：{args.cluster_config}", file=sys.stderr)
        return 2
    path = annotate(args.dataset, args.cluster_config, chunk_size=int(args.chunk_size))
    print(f"[annotate_clusters] DONE → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
