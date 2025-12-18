# Use an official Python runtime with PyTorch support
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Use --extra-index-url to allow pip to find torch-geometric in PyPI 
# while getting the light-weight torch from the CPU index.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Explicitly copy package init files to ensure they exist in the image
COPY worker/__init__.py ./worker/__init__.py
COPY src/__init__.py ./src/__init__.py

# Copy the rest of the application
COPY . .

ENV PYTHONUNBUFFERED=1