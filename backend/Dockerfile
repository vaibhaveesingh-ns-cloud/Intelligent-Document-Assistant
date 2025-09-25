FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal build toolchain for packages that may compile from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cargo \
    rustc \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements from the project root context
COPY backend/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
