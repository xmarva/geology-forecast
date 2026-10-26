# Ubuntu 24.04 with Python 3.10 and the CUDA 12.1 build of PyTorch 2.5.1.
# The CUDA runtime ships with the PyTorch wheel, so the host only needs an NVIDIA
# driver and the NVIDIA Container Toolkit (docker run --gpus all).
#
#   docker build -t geology-forecast .
#   docker run --rm -it --gpus all -p 8888:8888 --env-file .env \
#       -v "$(pwd)":/geology-forecast geology-forecast
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt-get install -y \
    python3-pip \
    git \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

ARG TORCH_VERSION=2.5.1+cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    --index-url ${TORCH_INDEX_URL}

# torch==2.5.1 in requirements.txt is already satisfied by the CUDA build above
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# The repository, with the competition data in data/, is mounted here at run time
WORKDIR /geology-forecast
EXPOSE 8888

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
