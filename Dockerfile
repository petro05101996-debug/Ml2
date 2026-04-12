FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt .
RUN pip install --prefer-binary --retries 5 -r requirements.txt

COPY . .
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8501

CMD ["sh", "/app/start.sh"]
