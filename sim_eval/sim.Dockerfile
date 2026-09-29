# ==============================================================================
# sim_eval/sim.Dockerfile
# NoMaD closed-loop simulation evaluation — iGibson 2.x, headless, EGL rendering.
#
# Builds ON TOP of nomad:latest so the simulator and the model share one conda
# environment (vint_train, Python 3.8.5). That matters from P1 onwards: the
# bridge runs the sim and the policy in a single process, so they cannot live
# in separate envs. iGibson declares no torch/numpy pins that clash with the
# NoMaD stack (torch 2.4.1+cu121, numpy 1.24.3) — verified before adopting it.
#
# Host requirements:
#   - NVIDIA driver 550.54.14, NVIDIA Container Toolkit
#   - nomad:latest already built (see ../nomad.Dockerfile)
#
# Build:
#   docker build -f sim_eval/sim.Dockerfile -t nomad_sim:latest .
#
# Run: ./sim_eval/sim_up.sh   (creates the screen session + container)
# ==============================================================================

FROM nomad:latest

ENV DEBIAN_FRONTEND=noninteractive


# ------------------------------------------------------------------------------
# LAYER 1 — OS packages for the iGibson renderer
#
# cmake + the inherited g++/nvcc build iGibson's C++ MeshRenderer at pip-install
# time (setup.py declares a CMakeExtension, so this is not optional).
#
# libegl1/libglvnd0 are the GLVND *loader*. The CUDA base image ships only the
# vendor driver (libEGL_nvidia.so.0) and no loader, so without these EGL cannot
# be dlopen'd at all. The X11 libs (libxrandr/xinerama/cursor/xi) are pulled in
# by iGibson's bundled GLFW sources at link time even for a headless build.
#
# The *-dev packages are the compile-time half and are equally required: the
# renderer includes CUDA's cuda_gl_interop.h, which includes <GL/gl.h>. Without
# mesa-common-dev/libgl1-mesa-dev that header does not exist and the build dies
# partway through EGLRendererContext. (Mesa supplies only the headers here —
# at runtime GLVND still dispatches to the NVIDIA driver, see LAYER 2.)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libegl1 \
        libglvnd0 \
        libgles2 \
        libopengl0 \
        libglu1-mesa \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        mesa-common-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------------------------
# LAYER 2 — Point GLVND at the NVIDIA EGL driver
#
# libglvnd picks a vendor by reading /usr/share/glvnd/egl_vendor.d/*.json in
# name order. Ubuntu ships only 50_mesa.json, so EGL would silently fall back to
# Mesa software rendering (llvmpipe) — slow, and not the GPU path this phase is
# meant to prove. The NVIDIA Container Toolkit injects the driver .so but not
# this manifest, so we write it ourselves; "10_" sorts ahead of "50_mesa".
# ------------------------------------------------------------------------------
RUN printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json


# ------------------------------------------------------------------------------
# LAYER 3 — iGibson
#
# 2.2.2 is the final release of the iGibson 2.x line named in the plan (D1).
# Pinned so a rebuild six months from now produces the same simulator.
# This layer is slow (~5-10 min): pip compiles the MeshRenderer with cmake.
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir igibson==2.2.2


# ------------------------------------------------------------------------------
# LAYER 3b — Video encoding for P4's per-episode replays
#
# imageio-ffmpeg ships a static ffmpeg binary, which is the whole point: the
# episodes are encoded headless inside this container, with no system ffmpeg to
# install and no apt package to drift. matplotlib (the panels) and Pillow (the
# frames) are already here, inherited from nomad:latest.
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir "imageio>=2.19,<3" "imageio-ffmpeg>=0.4.7"


# ------------------------------------------------------------------------------
# LAYER 4 — Runtime environment
#
# Assets are NOT baked into the image: scene meshes are tens of GB and belong on
# the shared disk, mounted at /igibson_data. iGibson resolves each dataset root
# from these env vars (igibson/__init__.py), overriding its global_config.yaml.
#
# NVIDIA_DRIVER_CAPABILITIES=all is what makes the driver's GL/EGL libraries
# visible inside the container; the default "compute,utility" hides them and
# rendering fails before it starts. It is set here as well as on `docker run`
# so the image is correct even if someone runs it by hand.
# ------------------------------------------------------------------------------
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    GIBSON_ASSETS_PATH=/igibson_data/assets \
    GIBSON_DATASET_PATH=/igibson_data/g_dataset \
    IGIBSON_DATASET_PATH=/igibson_data/ig_dataset \
    KEY_PATH=/igibson_data/igibson.key


LABEL maintainer="f.nazlias@gmail.com" \
      project="NoMaD closed-loop sim evaluation" \
      base="nomad:latest" \
      simulator="iGibson 2.2.2"
