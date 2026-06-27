# recon-sfm: COLMAP + GLOMAP (CUDA, sm_120) + classic CV / geometry stack.
# Torch-free on purpose; the deep-learning stack lives in recon-dl (FROM this).
#
# Override the CUDA base or target arch at build time, e.g.:
#   docker build --build-arg CUDA_TAG=12.8.1-devel-ubuntu24.04 --build-arg CUDA_ARCH=120 ...
ARG CUDA_TAG=12.9.1-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
ARG CUDA_ARCH=120

# ---- system + COLMAP/GLOMAP build deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates build-essential cmake ninja-build pkg-config \
      ffmpeg exiftool \
      libboost-program-options-dev libboost-graph-dev libboost-system-dev \
      libboost-filesystem-dev \
      libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
      libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev \
      libcgal-dev libceres-dev libcurl4-openssl-dev \
      libopenimageio-dev openimageio-tools libopencv-dev \
      libgl1 libglib2.0-0 \
      python3 python3-dev python3-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

# ---- isolated python venv (avoids PEP 668 / system python) ----
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv $VIRTUAL_ENV \
 && pip install --no-cache-dir --upgrade pip wheel setuptools

# ---- COLMAP from source, CUDA enabled, built for sm_120 ----
# GUI off => headless, no Qt needed. CMAKE_CUDA_ARCHITECTURES=120 is the fix for the
# 'compute_native'/'compute_120' failures on Blackwell (colmap/colmap#3408).
RUN git clone --depth 1 --branch 4.0.4 https://github.com/colmap/colmap.git /opt/colmap \
 && cmake -S /opt/colmap -B /opt/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGUI_ENABLED=OFF \
 && cmake --build /opt/colmap/build --target install \
 && rm -rf /opt/colmap/build

# ---- GLOMAP (global SfM mapper, links against the COLMAP just installed) ----
RUN git clone --depth 1 --branch 1.2.0 https://github.com/colmap/glomap.git /opt/glomap \
 && cmake -S /opt/glomap -B /opt/glomap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
 && cmake --build /opt/glomap/build --target install \
 && rm -rf /opt/glomap/build

# ---- python geometry / CV libs (NO torch here) ----
RUN pip install --no-cache-dir \
      numpy scipy \
      opencv-contrib-python-headless \
      open3d \
      pycolmap \
      pyyaml tqdm pillow matplotlib

# ---- TEASER++ global registration (optional; Open3D RANSAC/FGR is the fallback) ----
RUN git clone --depth 1 https://github.com/MIT-SPARK/TEASER-plusplus.git /opt/teaserpp \
 && cmake -S /opt/teaserpp -B /opt/teaserpp/build \
      -DTEASERPP_PYTHON_VERSION=3.12 -DBUILD_TESTS=OFF -DBUILD_DOC=OFF \
 && cmake --build /opt/teaserpp/build --target teaserpp_python -j"$(nproc)" \
 && pip install --no-cache-dir /opt/teaserpp/build/python \
 ; echo "(teaser++ optional: continuing regardless of build result)"

WORKDIR /work
