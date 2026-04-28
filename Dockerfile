FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

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
RUN pip install --no-cache-dir -r requirements-serverless.txt

RUN pip install --no-cache-dir --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio==2.5.1 torchvision==0.20.1

COPY app/__init__.py app/__init__.py
COPY app/pipelines.py app/pipelines.py
COPY app/store.py app/store.py
COPY handler.py handler.py
COPY wishperx.py wishperx.py

CMD ["python", "-u", "/workspace/handler.py"]
