# Use an NVIDIA CUDA runtime base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# # Install Python, pip, git, and other dependencies
# RUN apt-get update && \
#     apt-get install -y python3-pip git ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y wget

RUN mkdir -p /home/user

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/user/miniconda.sh && \
    bash /home/user/miniconda.sh -b -p /opt/conda && \
    rm /home/user/miniconda.sh && \
    /opt/conda/bin/conda clean -afy


# Create and activate a conda environment (single RUN layer)
RUN /bin/bash -c "\
    source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n nnInteractive python=3.12 -c conda-forge && \
    conda activate nnInteractive && \
    conda clean -afy \
"

RUN /opt/conda/bin/conda install pip

RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip install nninteractive

WORKDIR /opt/server

COPY samurai_server.py samurai_server.py
COPY requirements.txt requirements.txt

RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip install -r requirements.txt

RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip  list

# # Upgrade pip
# RUN python3 -m pip install --upgrade pip

# # Install PyTorch with CUDA 12.6 support (as per nnInteractive instructions)
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# # Clone the nnInteractive repository into /opt/nnInteractive
# RUN git clone https://github.com/MIC-DKFZ/nnInteractive.git /opt/nnInteractive

# Install nnInteractive in editable mode
# RUN pip install -e /opt/nnInteractive

# Set working directory
# WORKDIR /opt/nnInteractive

RUN useradd -m user && chown -R user:user /opt/server

USER user

# Default command: open an interactive Python shell
CMD ["/opt/conda/envs/nnInteractive/bin/python3.12", "samurai_server.py"]
