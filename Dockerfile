FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serverless.txt requirements-serverless.txt

RUN pip install --no-cache-dir --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio==2.5.1 torchvision==0.20.1

RUN pip install --no-cache-dir --no-deps -r requirements-serverless.txt

COPY app/__init__.py app/__init__.py
COPY app/pipelines.py app/pipelines.py
COPY app/store.py app/store.py
COPY handler.py handler.py
COPY wishperx.py wishperx.py

CMD ["python", "-u", "/workspace/handler.py"]
