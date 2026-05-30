# Use python:3.12-slim as a base image for a CPU-only environment.
# NOTE: This image does NOT include NVIDIA drivers or the CUDA toolkit.
# JAX will run on CPU only. If GPU support is needed, consider using
# a base image from NVIDIA, e.g., nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM python:3.12-slim

# Prevent Python from writing .pyc files and from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies. This is done early to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project's "library" code into the image.
# We explicitly copy directories to exclude `mainhydra.py` and `config/`.
COPY data /app/data/
COPY inference /app/inference/
COPY losses /app/losses/
COPY metrics /app/metrics/
COPY models /app/models/
COPY postprocessing /app/postprocessing/
COPY training /app/training/
COPY utils /app/utils/

# Add the project root to PYTHONPATH. This allows Python to find your modules.
ENV PYTHONPATH="/app"

# Drop into a bash shell when the container starts.
# This allows the user to run commands interactively.
CMD ["bash"]