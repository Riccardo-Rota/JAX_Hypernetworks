FROM python:3.12-slim

# Prevent Python from writing .pyc files and from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy your project's "library" code into the image
# We explicitly copy directories to exclude `main.py` and `config/`
COPY . /app

# Add the project root to PYTHONPATH. This allows Python to find your modules.
ENV PYTHONPATH="/app"

# Drop into a bash shell when the container starts.
# This allows the user to run commands interactively.
CMD ["bash"]