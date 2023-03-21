FROM python:3.8.16-slim-bullseye
LABEL MAINTAINER="adamadaniel321@gmail.com"


ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update -y
RUN apt-get install -y --fix-missing \
    git \
    apt-get install build-essential \
    cmake \
    wget \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python-dev \
    python-pip \
    python3-dev

## Compile Dlib
RUN wget http://dlib.net/files/dlib-19.9.tar.bz2
RUN tar xvf dlib-19.9.tar.bz2
RUN cd dlib-19.9 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    cmake --build . --config Release && \
    make install && \
    sudo ldconfig && \
    cd .. && \
    pkg-config --libs --cflags dlib-1 && \
    cd dlib-19.9 && \
    python3 setup.py install

WORKDIR /app
COPY ["config","requirements.txt", "face_verification", "static", "templates", ".env", "app.py", "encodings.pickle", "face_database", "./"]
RUN pip3 install -r /app/requirements.txt
# EXPOSE 8080
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:8080", "app:app", "--workers=5"]