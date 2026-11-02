FROM python:3.10-slim

WORKDIR /app

# OpenCV 및 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Render가 지정하는 기본 포트($PORT) 사용 환경 변수 대응
CMD uvicorn server:app --host 0.0.0.0 --port $PORT