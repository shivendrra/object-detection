FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION=4.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev \
    python3-dev python3-pip python3-numpy \
    ffmpeg wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install numpy

# Build OpenCV with CUDA
WORKDIR /opt
RUN git clone https://github.com/opencv/opencv.git --branch ${OPENCV_VERSION} --depth 1
RUN git clone https://github.com/opencv/opencv_contrib.git --branch ${OPENCV_VERSION} --depth 1

RUN mkdir /opt/opencv/build
WORKDIR /opt/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D BUILD_EXAMPLES=OFF ..

RUN make -j$(nproc) && make install && ldconfig

# Set working directory and copy scripts
WORKDIR /app
COPY script.py detection.py run.py /app/

CMD ["python3", "script.py", "run.py"]
