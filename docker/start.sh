#!/usr/bin/env sh
set -eu

echo "[start] python version:"
python --version
echo "[start] PORT=${PORT:-8501}"

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
