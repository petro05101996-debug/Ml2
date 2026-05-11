from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelQualityGate:
    status: str
    wape: Optional[float] = None
    naive_wape: Optional[float] = None
    improvement_vs_naive_pct: Optional[float] = None
    rolling_wape_max: Optional[float] = None
    bias_pct: Optional[float] = None
    std_ratio: Optional[float] = None
    prediction_variability: Optional[float] = None
    stockout_share: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_model_quality_gate(metrics: Dict[str, Any]) -> ModelQualityGate:
    reasons: List[str] = []
    warnings: List[str] = []
    blockers: List[str] = []
    def f(name: str):
        try:
            val = float(metrics.get(name))
            return val if val == val and abs(val) != float("inf") else None
        except Exception:
            return None
    wape = f("wape")
    naive_wape = f("naive_wape") or f("best_naive_wape")
    improvement = f("improvement_vs_naive_pct") or f("naive_improvement_pct")
    stockout_share = f("stockout_share")
    status = "production_allowed"
    if wape is None:
        status = "blocked"; blockers.append("wape_unknown")
    elif wape > 45:
        status = "blocked"; blockers.append("wape_above_45")
    elif wape > 35:
        status = "experimental_only"; warnings.append("wape_35_45")
    elif wape > 25:
        status = "controlled_test_only"; warnings.append("wape_25_35")
    if improvement is not None and improvement <= 0:
        status = max_status(status, "experimental_only")
        warnings.append("ml_not_better_than_naive")
    rolling = metrics.get("rolling_retrain_backtest") if isinstance(metrics.get("rolling_retrain_backtest"), dict) else {}
    rolling_verdict = str((rolling or {}).get("verdict", ""))
    if rolling_verdict in {"experimental_unstable", "unstable_experimental_only"} or rolling_verdict.startswith("experimental"):
        status = max_status(status, "experimental_only")
        warnings.append("rolling_retrain_unstable")
    elif rolling_verdict in {"test_only_unstable", "unstable_test_only"} or rolling_verdict.startswith("test_only"):
        status = max_status(status, "controlled_test_only")
        warnings.append("rolling_retrain_test_only")
    if stockout_share is not None:
        if stockout_share > 0.30:
            status = "blocked"; blockers.append("stockout_share_above_30")
        elif stockout_share > 0.15:
            status = max_status(status, "experimental_only"); warnings.append("stockout_share_15_30")
        elif stockout_share > 0.05:
            status = max_status(status, "controlled_test_only"); warnings.append("stockout_share_5_15")
    reasons.extend(warnings + blockers)
    return ModelQualityGate(status, wape, naive_wape, improvement, f("rolling_wape_max"), f("bias_pct"), f("std_ratio"), f("prediction_variability"), stockout_share, reasons, warnings, blockers)


def max_status(a: str, b: str) -> str:
    order = {"production_allowed": 0, "controlled_test_only": 1, "experimental_only": 2, "blocked": 3}
    reverse = {v: k for k, v in order.items()}
    return reverse[max(order.get(a, 0), order.get(b, 0))]
