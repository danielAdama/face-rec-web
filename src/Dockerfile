# FROM ubuntu:20.04
# RUN apt-get update && apt install -y python3-dev python3-pip python3.8.14-venv

FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
LABEL MAINTAINER="adamadaniel321@gmail.com"


ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y --fix-missing \
    git \
    apt-get install build-essential \
    pkg-config \
    cmake \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    libopenblas-dev \ 
    liblapack-dev

## Compile Dlib
RUN wget http://dlib.net/files/dlib-19.9.tar.bz2
RUN tar xvf dlib-19.9.tar.bz2
RUN cd dlib-19.9 && \
    mkdir build && \
    cd build && \
    cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 && \
    cmake --build . --config Release && \
    sudo make install && \
    sudo ldconfig && \
    cd .. && \
    pkg-config --libs --cflags dlib-1 && \
    cd dlib-19.9 && \
    python3 setup.py install

WORKDIR /app
COPY ["config","requirements.txt", "face_verification", "static", "templates", ".env", "app.py", "encodings.pickle", "face_database", "./"]
RUN pip3 install -r /app/requirements.txt
EXPOSE 8080
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "app:app", "--workers=5"]