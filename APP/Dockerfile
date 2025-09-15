# syntax=docker/dockerfile:1.4
# Dockerfile for FAICORD
# write by Jaedong, Oh (2025.05.18)
## Update by Jaedong, Oh (2025.07.04)

FROM python:3.12-slim-bullseye AS builder
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
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Inference Stage ---
FROM python:3.12-slim-bullseye
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
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
COPY . .
EXPOSE 8081
# CMD ["sleep", "infinity"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--reload"]