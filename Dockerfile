# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
RUN /root/.cargo/bin/uv pip install --no-cache -e .

# Create directory for model checkpoints
RUN mkdir -p /models/open_clip

# Optional: Copy model checkpoints
# COPY models/open_clip/*.ckpt /models/open_clip/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "tulip_api.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
