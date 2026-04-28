# ────────────────────────────────────────────────────────
# Mammography Inference (Dockerized) – CUDA 12.1 + Python 3.10
# ────────────────────────────────────────────────────────

# Base image: NVIDIA CUDA Runtime + Ubuntu 22.04
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set venv, and install system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev build-essential libglib2.0-0 libgl1-mesa-glx python3-gdcm && \
    rm -rf /var/lib/apt/lists/*

# Create a symlink: python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Setup virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Install base packages
COPY requirements.txt .

RUN pip install --upgrade pip wheel && \
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg && \
    pip install -r requirements.txt


# Copy source code
COPY src ./src
COPY entrypoint.sh /workspace/entrypoint.sh

# Make EP shell executable
RUN chmod +x /workspace/entrypoint.sh

# Custom docker command
ENTRYPOINT ["/workspace/entrypoint.sh", "python", "src/predict.py"]

# Default command (previous)
# ENTRYPOINT ["python", "-W", "ignore", "src/predict.py"]

CMD ["--help"]
