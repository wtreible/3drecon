# recon-dl: deep-learning reconstruction stack on top of recon-sfm.
# Adds PyTorch (cu128, sm_120-capable), gsplat, nerfstudio (splatfacto), hloc, VGGT.
ARG SFM_IMAGE=recon-sfm:latest
FROM ${SFM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# Build CUDA extensions (gsplat) for Blackwell sm_120.
ARG TORCH_CUDA_ARCH_LIST=12.0
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TCNN_CUDA_ARCHITECTURES=120
# Parallelize CUDA-extension compiles (gsplat has ~40 heavy sm_120 kernels). Without this,
# torch's ninja build runs ONE file at a time -> multi-hour build. 8 jobs fits 45GB RAM.
ENV MAX_JOBS=8

# ---- PyTorch with cu128 wheels (>=2.7 required for sm_120; 2.11 current) ----
RUN pip install --no-cache-dir ninja \
 && pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128

# ---- gsplat (compiles rasterization kernels for sm_120) ----
# Pin 1.4.0: it's exactly what nerfstudio 1.1.5 requires AND it predates gsplat 1.5's
# 3DGUT "ParallelBatch" kernels, whose single-file cicc compile explodes (>1h) on
# sm_120 under -O3. --no-build-isolation builds against the cu128 torch above (a fresh
# isolated torch would be CUDA-13 and mismatch the 12.9 toolkit). MAX_JOBS parallelizes.
RUN pip install --no-cache-dir --no-build-isolation gsplat==1.4.0

# ---- nerfstudio (splatfacto uses gsplat; no tiny-cuda-nn needed) ----
RUN pip install --no-cache-dir nerfstudio

# ---- hloc + learned matchers (SuperPoint/LightGlue/ALIKED/DISK) — needs torch ----
RUN pip install --no-cache-dir \
      git+https://github.com/cvg/Hierarchical-Localization.git

# ---- VGGT (feed-forward SfM/pose/depth) ----
RUN git clone --depth 1 https://github.com/facebookresearch/vggt.git /opt/vggt \
 && pip install --no-cache-dir -e /opt/vggt

# nerfstudio / hloc / vggt may drag in a CPU torch via their pins — reassert cu128 build last.
RUN pip install --no-cache-dir --force-reinstall --no-deps torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128

# torch>=2.6 flipped torch.load to weights_only=True; nerfstudio 1.1.5 predates that and
# fails to load its OWN checkpoints (export/render/resume). These are trusted local files,
# so re-enable full unpickling for nerfstudio's loads.
RUN NS=/opt/venv/lib/python3.12/site-packages/nerfstudio && \
    sed -i 's/map_location="cpu")/map_location="cpu", weights_only=False)/g' \
      $NS/engine/trainer.py $NS/utils/eval_utils.py \
      $NS/scripts/downloads/download_data.py

# Persist caches on the host (the repo is bind-mounted at /work, so /work/data survives
# --rm). Crucially TORCH_EXTENSIONS_DIR: gsplat 1.4.0 JIT-compiles its sm_120 kernels on
# first use (~8 min); persisting them means that happens exactly once, not every run.
ENV TORCH_HOME=/work/data/torch_cache \
    HF_HOME=/work/data/hf_cache \
    TORCH_EXTENSIONS_DIR=/work/data/torch_extensions
WORKDIR /work
