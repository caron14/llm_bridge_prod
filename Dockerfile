# Base image with CUDA support
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    git-lfs \
    curl \
    vim \
    openssh-client \
    libopenmpi-dev \
    openmpi-bin \
    libhwloc-dev \
    libnuma-dev \
    librdmacm-dev \
    libibverbs-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# Create conda environment
ENV CONDA_PATH=/opt/conda_env
RUN conda create --prefix $CONDA_PATH python=3.11 -y

# Activate the environment and set environment variables
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    conda config --set auto_activate_base false

# Install CUDA toolkit and cuDNN through conda
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    conda install -y cuda-toolkit=12.4.1 -c nvidia/label/cuda-12.4.1 && \
    conda install -y -c conda-forge cudnn && \
    conda install -y gcc_linux-64 gxx_linux-64 && \
    conda install -y git && \
    conda install -y anaconda::git-lfs

# Set up environment variables for conda environment
RUN mkdir -p $CONDA_PATH/etc/conda/activate.d && \
    echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDNN_PATH=$CUDNN_PATH' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CUDA_HOME=$CUDA_HOME' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export ORIGINAL_CONDA_PATH=$CONDA_PATH' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:'$CONDA_PATH'/lib:'$CONDA_PATH'/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDNN_PATH='$CONDA_PATH'/lib' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CUDA_HOME='$CONDA_PATH'/' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export CONDA_PATH='$CONDA_PATH'/' >> $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh && \
    chmod +x $CONDA_PATH/etc/conda/activate.d/edit_environment_variable.sh

# Initialize git lfs
RUN git lfs install

# Upgrade pip and install basic tools
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    pip install --upgrade pip wheel cmake ninja

# Clone repositories
WORKDIR /workspace
RUN mkdir -p deps && \
    cd deps && \
    git clone https://github.com/volcengine/verl.git && \
    git clone https://github.com/NVIDIA/apex && \
    git clone https://github.com/NVIDIA/TransformerEngine

# Install VERL
WORKDIR /workspace/deps/verl
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh && \
    pip install --no-deps -e .

# Install Python dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    pip install --no-cache-dir \
    six regex numpy==1.26.4 deepspeed wandb huggingface_hub tensorboard \
    mpi4py sentencepiece nltk ninja packaging wheel transformers accelerate \
    safetensors einops peft datasets trl matplotlib sortedcontainers brotli \
    zstandard cryptography colorama audioread soupsieve defusedxml babel \
    codetiming zarr tensorstore pybind11 scikit-learn nest-asyncio httpcore \
    pytest pylatexenc tensordict pyzmq==27.0 tensordict==0.9.1 ipython \
    hydra-core && \
    pip install -U "ray[data,train,tune,serve]" && \
    pip install --upgrade protobuf

# Install Apex
WORKDIR /workspace/deps/apex
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    pip cache purge && \
    python setup.py install \
    --cpp_ext --cuda_ext \
    --distributed_adam \
    --deprecated_fused_adam \
    --xentropy \
    --fast_multihead_attn

# Install Flash Attention 2
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    MAX_JOBS=8 pip install flash-attn==2.6.3 --no-build-isolation

# Install TransformerEngine
WORKDIR /workspace/deps/TransformerEngine
RUN git submodule update --init --recursive && \
    git checkout release_v2.4 && \
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate $CONDA_PATH && \
    NMAX_JOBS=8 VTE_FRAMEWORK=pytorch pip install --no-cache-dir .

# Clone the main repository
WORKDIR /workspace
RUN git clone https://github.com/matsuolab/llm_bridge_prod.git

# Create verification script
RUN cat > /workspace/verify_installation.py << 'EOF'
import importlib, apex, torch, sys

# Check if modules can be imported
for mod in (
    "apex.transformer",
    "apex.normalization.fused_layer_norm",
    "apex.contrib.optimizers.distributed_fused_adam",
    "flash_attn",
    "verl.trainer",
    "ray",
    "transformer_engine",
):
    print("✅" if importlib.util.find_spec(mod) else "❌", mod)

# Check flash-attention version
try:
    import flash_attn
    flash_ver = getattr(flash_attn, "__version__", "unknown")
except ImportError:
    flash_ver = "not installed"

# Check if verl.trainer.main_ppo exists
try:
    from verl.trainer import main_ppo as _main_ppo
    main_ppo_flag = "✅ main_ppo in verl.trainer"
except ImportError:
    main_ppo_flag = "❌ main_ppo in verl.trainer"
print(main_ppo_flag)

# Check Ray version
try:
    import ray
    ray_ver = getattr(ray, "__version__", "unknown")
except ImportError:
    ray_ver = "not installed"

# Check TransformerEngine version
try:
    import transformer_engine
    te_ver = getattr(transformer_engine, "__version__", "unknown")
except ImportError:
    te_ver = "not installed"

# Print version information
print("Flash-Attention ver.:", flash_ver, end=" | ")
print("Ray ver.:", ray_ver, end=" | ")
print("TransformerEngine ver.:", te_ver, end=" | ")
print("Apex ver.:", getattr(apex, "__version__", "unknown"),
      "| Torch CUDA:", torch.version.cuda,
      "| Python:", sys.version.split()[0])
EOF

# Set up entrypoint to activate conda environment
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /entrypoint.sh && \
    echo 'conda activate '$CONDA_PATH >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Add labels
LABEL maintainer="LLM Bridge Competition 2025"
LABEL description="Docker image for LLM training with VERL, Apex, Flash Attention, and TransformerEngine"
LABEL cuda.version="12.4.1"
LABEL python.version="3.11"