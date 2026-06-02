FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install torch first (420 MB) in its own layer so it gets cached independently
RUN pip install --no-cache-dir --timeout=600 torch==2.11.0
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

CMD ["python", "-m", "vision.camera"]
