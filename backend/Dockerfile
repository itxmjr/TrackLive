# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and building 'lap'
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
# Note: Since we use pyproject.toml, we can use pip to install from it
COPY backend/pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the backend code
COPY backend/src /app/backend/src
COPY backend/data /app/backend/data
COPY backend/yolov8n.pt /app/backend/yolov8n.pt

# Expose the port that Hugging Face Spaces expects (7860)
EXPOSE 7860

# Command to run the application
# We use --host 0.0.0.0 and --port 7860 as required by HF Spaces
CMD ["uvicorn", "backend.src.api:app", "--host", "0.0.0.0", "--port", "7860"]
