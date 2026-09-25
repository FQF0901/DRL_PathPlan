"""奖励模型（N2 契约 §3）：可插拔奖励项 + 聚合 + 评估 KPI。

对外 API
--------
- 注册表：``register_term`` / ``make_term`` / ``build_terms`` / ``available_terms`` / ``Term``
- 奖励项：``DEFAULT_TERM_CONFIGS``（安全 / 舒适 / 效率 / 合规 / 达成）
- 聚合：``RewardAggregator`` / ``AggregationConfig`` / ``StepReward`` / ``CarlRule`` /
  ``ShapingDecay`` / ``assign_credits`` / ``grouped_discounted_returns``
- KPI：``compute_kpis`` / ``compute_kpis_by_primary`` / ``wilson_ci`` / ``Thresholds`` /
  ``episode_kpi`` / ``KPI_NAMES``

纯 Python/NumPy；不 import env/metadrive，导入无副作用。
"""

from __future__ import annotations

from reward_model.aggregation import (
    AggregationConfig,
    CarlRule,
    RewardAggregator,
    ShapingDecay,
    StepReward,
    assign_credits,
    default_carl_rules,
    default_terminal_values,
    discounted_return,
    grouped_discounted_returns,
)
from reward_model.kpi import (
    KPI_NAMES,
    Thresholds,
    compute_kpis,
    compute_kpis_by_primary,
    episode_kpi,
    incidental_labels,
    overall_success_target,
    per_category_target,
    primary_label,
    wilson_ci,
)
from reward_model.registry import (
    Term,
    TermConfig,
    available_terms,
    build_terms,
    get_term_class,
    make_term,
    register_term,
)
from reward_model.terms import DEFAULT_TERM_CONFIGS

__all__ = [
    # registry
    "Term",
    "TermConfig",
    "register_term",
    "available_terms",
    "get_term_class",
    "make_term",
    "build_terms",
    # terms
    "DEFAULT_TERM_CONFIGS",
    # aggregation
    "AggregationConfig",
    "CarlRule",
    "RewardAggregator",
    "ShapingDecay",
    "StepReward",
    "assign_credits",
    "default_carl_rules",
    "default_terminal_values",
    "discounted_return",
    "grouped_discounted_returns",
    # kpi
    "KPI_NAMES",
    "Thresholds",
    "compute_kpis",
    "compute_kpis_by_primary",
    "episode_kpi",
    "incidental_labels",
    "overall_success_target",
    "per_category_target",
    "primary_label",
    "wilson_ci",
]
