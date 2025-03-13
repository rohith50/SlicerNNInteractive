# Use an NVIDIA CUDA runtime base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, git, and other dependencies
RUN apt-get update && \
    apt-get install -y python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.6 support (as per nnInteractive instructions)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Clone the nnInteractive repository into /opt/nnInteractive
RUN git clone https://github.com/MIC-DKFZ/nnInteractive.git /opt/nnInteractive

# Install nnInteractive in editable mode
RUN pip install -e /opt/nnInteractive

# Set working directory
WORKDIR /opt/nnInteractive

USER user

# Default command: open an interactive Python shell
CMD ["python3", "samurai_server.py"]