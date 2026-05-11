from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RunMetadata:
    run_id: str
    dataset_hash: str
    model_version: str
    git_sha: str
    app_version: str
    created_at: str
    selected_sku: str
    scenario_mode: str
    decision_status: str
    model_quality: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def build_run_metadata(
    *,
    dataset_signature: Any,
    selected_sku: str = "unknown",
    scenario_mode: str = "unknown",
    decision_status: str = "unknown",
    model_quality: Optional[Dict[str, Any]] = None,
    data_quality: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    blockers: Optional[List[str]] = None,
    model_version: str = "decision-analyst-v1",
    app_version: str = "production-v1-contract",
) -> RunMetadata:
    dataset_hash = stable_hash(dataset_signature)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = stable_hash({"dataset_hash": dataset_hash, "created_at": created_at, "sku": selected_sku, "mode": scenario_mode})
    return RunMetadata(
        run_id=run_id,
        dataset_hash=dataset_hash,
        model_version=model_version,
        git_sha=current_git_sha(),
        app_version=app_version,
        created_at=created_at,
        selected_sku=str(selected_sku),
        scenario_mode=str(scenario_mode),
        decision_status=str(decision_status),
        model_quality=model_quality or {},
        data_quality=data_quality or {},
        warnings=list(warnings or []),
        blockers=list(blockers or []),
    )
