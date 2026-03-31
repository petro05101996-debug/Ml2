# ML Pricing App Agent Guide

## How to run
- Install deps: `pip install -r requirements.txt`
- Start UI: `streamlit run app.py`
- Run tests: `pytest -q`

## Definition of done
- No future leakage in feature/data preparation (no backward fill from future observations).
- Objective mode changes optimization behavior.
- What-if multipliers materially affect output metrics.
- Raw/adjusted/baseline metrics are explicitly separated in outputs.
- Tests pass locally.

## Parts that must not be broken
- Universal CSV normalization (`data_adapter.py`).
- Scenario runner/sensitivity outputs (`what_if.py`).
- Recommendation narrative consistency (`recommendation.py`).

## Non-hidden assumptions
- `cost` may be proxied as `0.65 * price` when absent.
- Confidence is heuristic and should be shown as advisory, not guarantee.
- Small datasets produce unstable recommendations and must emit warnings.
