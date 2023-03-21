FROM python:3.8.16-slim-bullseye
LABEL MAINTAINER="adamadaniel321@gmail.com"

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y --fix-missing \
    git \
    build-essential \
    cmake \
    wget \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev

## Compile Dlib
COPY dlib-19.9 /root
WORKDIR /root
RUN cd dlib-19.9 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    make install && \
    ldconfig && \
    cd .. && \
    pkg-config --libs --cflags dlib-1 && \
    cd dlib-19.9 && \
    python3 setup.py install

WORKDIR /root/app
COPY . /root/app
RUN pip3 install -r /app/requirements.txt
# EXPOSE 8080
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "app:app", "--workers=5"]