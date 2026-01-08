# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files to disc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system dependencies (if any needed for CV/Audio libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for models
RUN mkdir -p models

# Default command (overridden in docker-compose)
CMD ["python", "app.py"]
