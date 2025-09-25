FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install only essential build and image libs (avoid OpenGL/Tesseract to prevent apt failures)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    python3-dev \
    pkg-config \
    libgomp1 \
    libjpeg62-turbo-dev \
    libpng-dev \
    zlib1g-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend requirements from the project root context
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
