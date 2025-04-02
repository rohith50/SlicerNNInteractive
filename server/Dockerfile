FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y wget

RUN useradd -m user 

RUN chown -R user:user /home/user

RUN mkdir -p /home/user/.config/matplotlib && \
    mkdir -p /home/user/.cache && \
    chown -R user:user /home/user/.config /home/user/.cache

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /home/user/miniconda.sh && \
    bash /home/user/miniconda.sh -b -p /opt/conda && \
    rm /home/user/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

RUN /bin/bash -c "\
    source /opt/conda/etc/profile.d/conda.sh && \
    conda create -y -n nnInteractive python=3.12 -c conda-forge && \
    conda activate nnInteractive && \
    conda clean -afy \
"

RUN /opt/conda/bin/conda install pip

RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

WORKDIR /opt/server

COPY nninteractive_slicer_server/main.py main.py
COPY requirements.txt requirements.txt

RUN /opt/conda/envs/nnInteractive/bin/python3.12 -m pip install -r requirements.txt

RUN chown -R user:user /opt/server

USER user

CMD ["/opt/conda/envs/nnInteractive/bin/python3.12", "main.py"]
