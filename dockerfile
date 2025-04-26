# FROM python:3.7

# ENV PORT 8082
# ENV HOSTDIR 0.0.0.0

# EXPOSE 8082

# RUN apt-get update -y && \
#     apt-get install -y python3-pip

# COPY ./requirements.txt /app/requirements.txt

# WORKDIR /app

# RUN pip install -r requirements.txt

# COPY . /app


# ENTRYPOINT ["python", "app.py"]


# Use a modern Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8082
ENV HOSTDIR 0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8082

# Run the application
CMD ["python", "app.py"]