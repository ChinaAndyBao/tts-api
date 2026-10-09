# syntax=docker/dockerfile:1
# Qwen3-TTS API 镜像（CPU 推理）
# 构建：docker build -t <REGISTRY>/tts-api:<TAG> .
# 说明：模型权重（4.3GB）不打进镜像，由 PVC 提供（见 K8S_DEPLOY.md）

FROM python:3.12-slim

# 系统依赖：sox（qwen-tts 音频处理）、libsndfile1（soundfile 读写 WAV）
RUN apt-get update && apt-get install -y --no-install-recommends \
    sox libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/tts-api

# 依赖分两层装：先 CPU 版 torch（国内走阿里云轮子源），再其余依赖
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0+cpu torchaudio==2.11.0+cpu \
      -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
    && pip install --no-cache-dir -r requirements.txt

# 服务代码 + 运维脚本（download_models.py 供 initContainer 按需拉权重）
COPY app.py README.md ./
COPY deploy/download_models.py deploy/smoke_test.py ./deploy/

RUN mkdir -p /root/tts-api/tasks /root/tts-api/voices /root/models

EXPOSE 9898
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9898", "--workers", "1"]
