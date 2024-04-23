# Select the base image
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash \
    CUDA_HOME=/usr/local/cuda-12.1 \
    HF_HOME=/common-repos/LMs \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"

RUN umask 000

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev python3-opencv libsqlite3-dev

## Install Ubuntu packages
RUN apt update && \
    apt -y upgrade && \
    apt install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        nginx \
        bash \
        dos2unix \
        git \
        ncdu \
        net-tools \
        openssh-server \
        libglib2.0-0 \
        libsm6 \
        libgl1 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        wget \
        curl \
        psmisc \
        rsync \
        vim \
        zip \
        unzip \
        htop \
        screen \
        tmux \
        pkg-config \
        libcairo2-dev \
        libgoogle-perftools4 \
        libtcmalloc-minimal4 \
        apt-transport-https \
        ca-certificates && \
    update-ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN mkdir /python
WORKDIR /python
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
RUN tar xzf Python-3.10.13.tgz
RUN rm Python-3.10.13.tgz
WORKDIR /python/Python-3.10.13
RUN ./configure --enable-optimizations
RUN make install

# link python 3.10 to python
RUN ln -s /usr/local/bin/python3 /usr/local/bin/python
RUN ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# Stage 2: Install LLaVA and python modules
FROM base as setup
RUN pip install --upgrade pip

# Install LLaVA
RUN git clone https://github.com/haotian-liu/LLaVA.git /LLaVA
WORKDIR /LLaVA
RUN pip install -e .
RUN pip install -e ".[train]"

RUN pip install torch==2.1.2 torchvision==0.16.2 triton==2.1.0 accelerate==0.26.1 deepspeed==0.13.1 pynvml==11.5.0 --upgrade
#RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# install sglang
#RUN git clone https://github.com/sgl-project/sglang.git /sglang
#WORKDIR /sglang
#RUN pip install -e "python[all]"

RUN pip install "sglang[all]"
RUN pip install wheel setuptools rtpt
RUN pip install flash-attn==2.5.2 --no-build-isolation
RUN pip install git+https://github.com/lapp0/outlines@object-json-schema-regex
RUN pip install -U datasets

# install jupiter stuff
RUN pip install jupyterlab ipywidgets==7.6.4 jupyter-ui-poll seaborn
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager@2.0
#RUN jupyter labextension enable @jupyter-widgets/jupyterlab-manager

# install transformers required for llava training -> might break sglang not tested
RUN pip install transformers==4.37.2

RUN mkdir /workspace
WORKDIR /workspace