# ==============================================================================
# nomad.Dockerfile
# NoMaD — Goal Masked Diffusion Policies (visualnav-transformer)
#
# Host requirements:
#   - NVIDIA driver 550.54.14 (supports CUDA 12.4)
#   - Docker >= 20.10
#   - NVIDIA Container Toolkit
#
# Build:
#   docker build -f nomad.Dockerfile -t nomad:latest .
#
# Run:
#   ./run_nomad.sh
#   or manually:
#   docker run --gpus all --shm-size=8g -it nomad:latest
# ==============================================================================


# ------------------------------------------------------------------------------
# LAYER 1 — Base image
# Use the official NVIDIA image that already has CUDA + cuDNN + compiler.
# - cuda 12.4.1   matches driver 550.54.14
# - cudnn9        required by PyTorch 2.x
# - devel         includes nvcc compiler (diffusion_policy needs it at install)
# - ubuntu22.04   LTS, well-supported
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Stop apt from asking timezone/keyboard questions during install
ENV DEBIAN_FRONTEND=noninteractive


# ------------------------------------------------------------------------------
# LAYER 2 — OS packages
# Cleanup (rm -rf) MUST be in the same RUN as the install.
# If it is a separate RUN, the files are already baked into the previous layer
# and the image size does not shrink.
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV / display libraries needed by the vision code
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        # General tools
        nano \
        vim \
        git \
        # Needed to download the Miniconda installer below
        wget \
        bzip2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------------------------
# LAYER 3 — Install Miniconda
# The NVIDIA base has no conda — we install it ourselves.
# Pinning the installer URL makes the build reproducible.
# conda clean -afy removes package caches we no longer need.
# ------------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -q \
        "https://repo.anaconda.com/miniconda/Miniconda3-py312_24.11.1-0-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && conda config --system --set auto_activate_base false \
    && conda clean -afy


# ------------------------------------------------------------------------------
# LAYER 4 — Clone the visualnav-transformer repo
# We clone inside the image so it is self-contained.
# --depth 1 skips the full git history (saves ~200MB).
# The local repo can be mounted over this at runtime for live editing.
# ------------------------------------------------------------------------------
WORKDIR /app

RUN git clone --depth 1 \
        https://github.com/robodhruv/visualnav-transformer.git \
        /app/visualnav-transformer


# ------------------------------------------------------------------------------
# LAYER 5 — Speed up conda with the libmamba solver
# The default conda solver can take 10-30 min on complex environments.
# libmamba solves the same problem in seconds to a few minutes.
# This must happen before conda env create.
# ------------------------------------------------------------------------------
RUN conda install -n base -c conda-forge conda-libmamba-solver -y \
    && conda config --system --set solver libmamba


# ------------------------------------------------------------------------------
# LAYER 6 — Create the vint_train conda environment
# This is the slowest layer (~2-5 min with libmamba).
# It is placed early so Docker caches it — code changes later in the file
# will not force this to rebuild.
# ------------------------------------------------------------------------------
RUN conda env create \
        -f /app/visualnav-transformer/train/train_environment.yml \
        -n vint_train \
    && conda clean -afy


# ------------------------------------------------------------------------------
# Activate vint_train for all remaining RUN steps AND for interactive sessions.
#
# Two separate things are needed:
#   1. SHELL  — makes future RUN build steps execute inside vint_train
#   2. ENV PATH — makes interactive "docker exec bash" sessions also find
#                 the right Python without manually running conda activate
# ------------------------------------------------------------------------------
ENV CONDA_DEFAULT_ENV=vint_train
ENV PATH="${CONDA_DIR}/envs/vint_train/bin:${PATH}"

SHELL ["conda", "run", "--no-capture-output", "-n", "vint_train", "/bin/bash", "-c"]


# ------------------------------------------------------------------------------
# LAYER 7 — Install the vint_train package itself
# The repo contains a pip-installable package inside train/.
# Without this step, "import vint_train" raises ModuleNotFoundError at runtime
# even though all the source files are present.
# -e = editable mode: changes to source are reflected immediately.
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir -e /app/visualnav-transformer/train/


# ------------------------------------------------------------------------------
# LAYER 8 — Extra pip packages
# wandb and huggingface_hub are NOT pinned — the original pins were outdated
# and caused conflicts with modern torch/protobuf.
# cffi and lmdb==1.4.1 are required for vint_dataset.py LMDB cache building.
# lmdb 2.x has a broken cffi backend on Python 3.8 + CUDA base image.
# --no-cache-dir keeps the layer smaller (no pip download cache kept).
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
        wandb \
        huggingface_hub \
        cffi \
        lmdb==1.4.1


# ------------------------------------------------------------------------------
# LAYER 9 — diffusion_policy
# Cloned at latest commit (pinned hash was removed — the original commit no
# longer exists in the upstream repo due to a rewritten history).
# Record the actual installed commit after build for reproducibility:
#   docker run --rm nomad:latest bash -c "cd /opt/diffusion_policy && git rev-parse HEAD"
# pip install -e . compiles C++ extensions — this works because we are on
# the "devel" base image which includes the CUDA compiler (nvcc).
# ------------------------------------------------------------------------------
RUN git clone https://github.com/real-stanford/diffusion_policy.git \
        /opt/diffusion_policy \
    && pip install --no-cache-dir -e /opt/diffusion_policy

# ------------------------------------------------------------------------------
# LAYER 10 — Entrypoint script
# Activates vint_train, then hands control to whatever command is given.
# "exec $@" is critical: it replaces this shell script with your command,
# making your process PID 1 so it receives docker stop signals cleanly
# (important for saving checkpoints before shutdown).
# ------------------------------------------------------------------------------
RUN printf '#!/bin/bash\nsource "${CONDA_DIR}/etc/profile.d/conda.sh"\nconda activate vint_train\nexec "$@"\n' \
        > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default: open an interactive shell (already inside vint_train)
CMD ["bash"]


# ------------------------------------------------------------------------------
# Image metadata
# ------------------------------------------------------------------------------
LABEL maintainer="you@institution.edu" \
      project="NoMaD" \
      base="nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04" \
      repo="https://github.com/robodhruv/visualnav-transformer"
