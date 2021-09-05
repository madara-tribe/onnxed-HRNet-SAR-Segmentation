FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION 3.7.0
ENV KERAS_VERSION 2.3.1
ENV TF_VERSION 2.3.0

WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev tree

RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall

RUN ln -s /usr/local/bin/python3.7 /bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /bin/pip3

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==${TF_VERSION} tensorflow-addons
RUN pip3 install keras==${KERAS_VERSION}
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install -U git+https://github.com/microsoft/onnxconverter-common
RUN pip3 install -U git+https://github.com/onnx/keras-onnx
CMD [ "/bin/bash" ]
