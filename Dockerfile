# 1. Base Image: Use a slim, official Python image to reduce final size.
# Replace 3.10 with your specific Python version.
FROM python:3.12.3-slim

# 2. Prevent Python from writing .pyc files and from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies (only if your Python packages require compilation, like psycopg2 or certain ML libraries)
# RUN apt-get update && apt-get install -y gcc build-essential && rm -rf /var/lib/apt/lists/*

# 5. Copy only the requirements file first. 
# This leverages Docker's layer caching. If requirements.txt hasn't changed, 
# Docker reuses the cached layer for the pip install, speeding up subsequent builds.
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code into the container
COPY . .

# 8. Define the default command to run your application
# Modify 'main.py' to whatever script acts as your entry point.
CMD ["python", "main.py"]