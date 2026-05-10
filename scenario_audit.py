"""Production audit helpers for reproducible scenario runs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    try:
        import numpy as np  # type: ignore

        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def stable_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_dataframe_payload(frame: Any, tail_rows: int | None = None) -> Dict[str, Any]:
    if not hasattr(frame, "columns"):
        return {"kind": "non_dataframe", "value": str(frame)}
    try:
        work = frame.copy()
        work.columns = [str(c) for c in work.columns]
        work = work.reindex(sorted(list(work.columns)), axis=1)
    except Exception:
        work = frame.copy()
    if tail_rows is not None:
        work = work.tail(int(tail_rows)).copy()
    for col in list(getattr(work, "columns", [])):
        try:
            if str(col).lower() == "date" or "date" in str(col).lower():
                import pandas as pd  # type: ignore

                parsed = pd.to_datetime(work[col], errors="coerce")
                work[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("")
            elif hasattr(work[col], "dtype") and str(work[col].dtype).startswith(("float", "Float")):
                work[col] = work[col].round(12)
        except Exception:
            pass
    try:
        records = work.where(work.notna(), None).to_dict(orient="records")
    except Exception:
        records = str(work)
    return {"kind": "dataframe", "columns": [str(c) for c in getattr(work, "columns", [])], "records": records}


def dataset_fingerprint_from_bundle(trained_bundle: Dict[str, Any]) -> Dict[str, Any]:
    daily = (trained_bundle or {}).get("daily_base")
    if not hasattr(daily, "columns"):
        value_hash = stable_sha256(str(daily))
        return {
            "dataset_hash": value_hash,
            "full_dataset_sha256": value_hash,
            "dataset_sample_hash": stable_hash(str(daily)),
            "rows_count": 0,
            "date_min": "",
            "date_max": "",
            "columns_hash": stable_sha256([]),
        }
    columns = [str(c) for c in daily.columns]
    full_payload = _canonical_dataframe_payload(daily)
    sample_payload = _canonical_dataframe_payload(daily, tail_rows=500)
    date_min = date_max = ""
    try:
        if "date" in daily.columns:
            import pandas as pd  # type: ignore

            dates = pd.to_datetime(daily["date"], errors="coerce").dropna()
            if len(dates):
                date_min = str(dates.min().date())
                date_max = str(dates.max().date())
    except Exception:
        pass
    full_hash = stable_sha256(full_payload)
    return {
        "dataset_hash": full_hash,
        "full_dataset_sha256": full_hash,
        "canonical_dataset_sha256": full_hash,
        "dataset_sample_hash": stable_hash(sample_payload),
        "rows_count": int(len(daily)),
        "date_min": date_min,
        "date_max": date_max,
        "columns_hash": stable_sha256(sorted(columns)),
    }


def dataset_hash_from_bundle(trained_bundle: Dict[str, Any]) -> str:
    return dataset_fingerprint_from_bundle(trained_bundle)["dataset_hash"]


def build_scenario_reproducibility_id(
    trained_bundle: Dict[str, Any],
    scenario_params: Dict[str, Any],
    mode: str,
    guardrail_mode: str,
    model_version: str,
    code_signature: str,
    feature_schema_version: str = "universal_csv_v1",
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    dataset_fingerprint = dataset_fingerprint_from_bundle(trained_bundle)
    dataset_hash = dataset_fingerprint["dataset_hash"]
    params_hash = stable_hash(scenario_params)
    catboost_bundle = (trained_bundle or {}).get("catboost_full_factor_bundle") or {}
    feature_cols = list(catboost_bundle.get("feature_cols", []) or [])
    config_hash = stable_hash({"config": config or {}, "feature_cols": feature_cols, "feature_schema_version": feature_schema_version})
    run_id = stable_hash(
        {
            "dataset_hash": dataset_hash,
            "dataset_sample_hash": dataset_fingerprint.get("dataset_sample_hash"),
            "rows_count": dataset_fingerprint.get("rows_count"),
            "date_min": dataset_fingerprint.get("date_min"),
            "date_max": dataset_fingerprint.get("date_max"),
            "columns_hash": dataset_fingerprint.get("columns_hash"),
            "scenario_params_hash": params_hash,
            "mode": mode,
            "guardrail_mode": guardrail_mode,
            "model_version": model_version,
            "code_signature": code_signature,
            "config_hash": config_hash,
            "feature_schema_version": feature_schema_version,
        }
    )
    return {
        "scenario_run_id": run_id,
        **dataset_fingerprint,
        "dataset_hash": dataset_hash,
        "model_version": model_version,
        "code_version": code_signature,
        "code_signature": code_signature,
        "config_hash": config_hash,
        "feature_schema_version": feature_schema_version,
        "scenario_params_hash": params_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": str(mode),
        "guardrail_mode": str(guardrail_mode),
    }
