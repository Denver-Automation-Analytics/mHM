FROM continuumio/miniconda3:latest

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    curl \
    wget \
    git \
    htop \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the mHM development conda environment from a declarative env file
COPY .devcontainer/env.yaml /tmp/env.yaml
RUN conda env create -f /tmp/env.yaml && conda clean -afy

# Make conda activate work in non-interactive shells and activate mhm-dev by default
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mhm-dev" >> /etc/bash.bashrc \
    && echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mhm-dev" >> /root/.bashrc

# Prepend the conda env bin to PATH so tools are available without explicit activation
ENV PATH="/opt/conda/envs/mhm-dev/bin:$PATH"

WORKDIR /workspace
