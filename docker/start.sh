#!/usr/bin/env sh
set -eu

echo "[start] python version:"
python --version

echo "[start] PORT=${PORT:-8501}"
echo "[start] working dir: $(pwd)"
echo "[start] files:"
ls -la

echo "[start] compile check..."
python -m py_compile app.py data_adapter.py data_schema.py what_if.py

echo "[start] import check..."
python - <<'PY'
mods = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "sklearn",
    "statsmodels",
    "openpyxl",
]
for m in mods:
    __import__(m)

try:
    __import__("catboost")
    print("[start] catboost import ok")
except Exception as e:
    print(f"[start] catboost import failed: {e}")
    raise

print("[start] imports_ok")
PY

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
