FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY scripts /app/scripts

RUN chmod +x /app/scripts/*.sh

ENV PORT=8000
ENV DEVICE=cpu
ENV MODEL_NAME=Ruicheng/moge-2-vitb-normal
ENV MODEL_DIR=/models/moge2
ENV HF_HOME=/models/.hf
ENV HUGGINGFACE_HUB_CACHE=/models/.hf/hub
ENV TORCH_HOME=/models/.torch
ENV JOBS_DIR=/app/data/jobs
ENV AUTO_DOWNLOAD_ON_START=false

EXPOSE 8000

CMD ["/app/scripts/start.sh"]
