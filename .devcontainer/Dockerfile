FROM continuumio/miniconda3:latest

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the mHM development conda environment
RUN conda create -y -n mhm-dev -c conda-forge \
    compilers \
    pkg-config \
    cmake \
    make \
    ninja \
    netcdf-fortran \
    fypp \
    git \
    && conda clean -afy

# Make conda activate work in non-interactive shells and activate mhm-dev by default
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mhm-dev" >> /etc/bash.bashrc \
    && echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mhm-dev" >> /root/.bashrc

# Prepend the conda env bin to PATH so tools are available without explicit activation
ENV PATH="/opt/conda/envs/mhm-dev/bin:$PATH"

WORKDIR /workspace
