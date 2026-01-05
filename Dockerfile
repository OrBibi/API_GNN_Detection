FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (Removed celery, Added rq)
RUN pip install --no-cache-dir \
    fastapi uvicorn rq redis python-multipart \
    torch torch-geometric pandas numpy scikit-learn seaborn matplotlib pyarrow fastparquet

# Copy the entire project
COPY . /app

# Create necessary directories
RUN mkdir -p uploads backend/static/results models

EXPOSE 8000

# Default command (overridden in docker-compose)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]