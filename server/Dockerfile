# Use an NVIDIA CUDA image as the base (for GPU support)
# FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV PATH="${PATH}:/home/user/.local/bin"

# Install basic tools and add deadsnakes PPA to install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set working directory
WORKDIR /home/user

# Copy requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r requirements.txt

# Install huggingface_hub for model downloading
RUN python3.11 -m pip install huggingface_hub

# Download the SAM2.1 model into /home/user/models and adjust permissions
RUN huggingface-cli download --local-dir /home/user/models apple/coreml-sam2.1-large && \
    chown -R 1000:1000 /home/user/models

# Copy your server and predictor code into the container
COPY --chown=1000:1000 samurai_server.py samurai_server.py
COPY --chown=1000:1000 sam2_numpy_predictor.py sam2_numpy_predictor.py

# Expose the FastAPI server port
EXPOSE 1526

# Switch to a non-root user for security (assuming UID 1000)
USER 1000

# Run the server
CMD ["python3.11", "samurai_server.py"]