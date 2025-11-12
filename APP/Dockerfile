# syntax=docker/dockerfile:1.4
# Dockerfile for FAICORD
# write by Jaedong, Oh (2025.05.18)
## Update by Jaedong, Oh (2025.11.12)

FROM python:3.11-slim-bullseye AS builder
WORKDIR /app

# --- Builder Stage ---
RUN echo "deb http://deb.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-venv \
        build-essential \
        git \
        locales \
        ffmpeg \
        wget \
        ca-certificates \
        curl \
        gnupg && \
    python -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# requirements + constraints 복사
COPY requirements.txt constraints.txt ./

# pip 업그레이드 및 설치
RUN pip install --upgrade pip wheel setuptools && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt -c constraints.txt

# --- Inference Stage ---
FROM python:3.11-slim-bullseye
WORKDIR /faicord

RUN echo "deb http://deb.debian.org/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-venv \
        build-essential \
        git \
        locales \
        ffmpeg \
        wget \
        ca-certificates \
        curl \
        gnupg && \
    python -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. 가상환경 복사 및 환경 설정
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# (선택) locale 설정
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["sleep", "infinity"]
