# MRMem Reproducibility Dockerfile
# CUDA 12.1 base image; PyTorch 2.5.1+cu121 is installed by pip below.
# (The NGC 23.10 image ships PyTorch 2.1 and would conflict with this pin.)

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Metadata
LABEL maintainer="Anonymous Authors"
LABEL description="MRMem: Hierarchical Memory Interfaces for Learned Retrieval"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace/mrmem

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    git \
    wget \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.1 support (must be installed first)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
RUN pip install --no-cache-dir \
    transformers==5.12.1 \
    accelerate==1.1.0 \
    deepspeed==0.16.4

# Additional dependencies
RUN pip install --no-cache-dir \
    safetensors==0.4.3 \
    tokenizers==0.22.1 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3 \
    datasets==4.5.0 \
    faiss-cpu==1.7.4 \
    rank-bm25==0.2.2 \
    scipy==1.11.3 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.1 \
    rouge-score==0.1.2 \
    tqdm==4.66.3 \
    pyyaml==6.0.1 \
    jinja2==3.1.2 \
    packaging==23.2

# Development tools (optional)
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    black==23.11.0 \
    flake8==6.1.0

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
RUN python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
RUN python -c "from transformers import AutoModel; print('Transformers import OK')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/workspace/mrmem/.cache/huggingface
ENV HF_HOME=/workspace/mrmem/.cache/huggingface

# Create cache directory
RUN mkdir -p /workspace/mrmem/.cache/huggingface

# Default command
CMD ["/bin/bash"]

# Build instructions:
# docker build -t mrmem:v1.0 -f Dockerfile .
#
# Run instructions:
# docker run --gpus all -it --rm \
#   -v $(pwd):/workspace/mrmem \
#   -v ~/.cache/huggingface:/workspace/mrmem/.cache/huggingface \
#   mrmem:v1.0
#
# For multi-GPU training:
# docker run --gpus all -it --rm \
#   --shm-size=32g \
#   -v $(pwd):/workspace/mrmem \
#   -v ~/.cache/huggingface:/workspace/mrmem/.cache/huggingface \
#   mrmem:v1.0
