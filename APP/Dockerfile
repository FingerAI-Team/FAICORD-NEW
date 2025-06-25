# syntax=docker/dockerfile:1.4
# Dockerfile for FAICORD  
# write by Jaedong, Oh (2025.05.18)
# --- Builder stage ---
FROM python:3.12-slim-bullseye AS builder
WORKDIR /app

# 필수 도구만 설치
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv build-essential git locales ffmpeg wget && \
    python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .

# 필요 패키지 설치
RUN --mount=type=secret,id=gh_token \
    git clone https://$(cat /run/secrets/gh_token)@github.com/Jaedong95/nsnet2-denoiser.git && \
    cd nsnet2-denoiser && pip install .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install git+https://github.com/wenet-e2e/wespeaker.git

# --- Inference image ---
FROM python:3.12-slim-bullseye
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /faicord

COPY --from=builder /opt/venv /opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# CMD ["sleep", "infinity"]