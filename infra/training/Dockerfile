FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY project/requirements.txt /requirements.txt
COPY project/requirements-dev.txt /requirements-dev.txt

RUN pip install --no-cache-dir --user -r /requirements.txt
RUN pip install --no-cache-dir --user -r /requirements-dev.txt