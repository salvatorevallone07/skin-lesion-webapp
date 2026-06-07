FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# Minimal system libs required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch to keep the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY inference.py train.py app.py ./

EXPOSE 8000

# model/ is mounted as a volume so trained artifacts survive container rebuilds
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
