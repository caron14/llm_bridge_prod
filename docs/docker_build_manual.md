# Docker Build Manual for LLM Bridge Environment

This manual provides step-by-step instructions for building and using the Docker image for the LLM Bridge Competition 2025 development environment.

## Prerequisites

1. **Docker Installation**
   - Docker Engine 20.10+ with BuildKit support
   - NVIDIA Container Toolkit for GPU support
   - At least 50GB of free disk space
   - 16GB+ RAM recommended for build process

2. **GPU Requirements**
   - NVIDIA GPU with CUDA 12.4 support
   - NVIDIA Driver 525.60.13 or higher

## Building the Docker Image

### Step 1: Prepare Build Environment

```bash
# Clone the repository if you haven't already
git clone git@github.com:matsuolab/llm_bridge_prod.git
cd llm_bridge_prod

# Ensure Dockerfile exists in the repository root
ls -la Dockerfile
```

### Step 2: Build the Image

**Option A: Standard Build (Recommended)**
```bash
# Build with default settings
docker build -t llm-bridge-env:latest .
```

**Option B: Build with Custom Settings**
```bash
# Build with more parallel jobs (if you have more CPU/RAM)
docker build \
  --build-arg MAX_JOBS=16 \
  -t llm-bridge-env:latest .

# Build with build cache (useful for debugging)
docker build \
  --progress=plain \
  -t llm-bridge-env:latest .
```

**Option C: Multi-stage Build with Cache**
```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
  --cache-from llm-bridge-env:latest \
  -t llm-bridge-env:latest .
```

### Step 3: Monitor Build Progress

The build process will take 30-60 minutes depending on your hardware. Key stages:
1. Base image download (5-10 min)
2. System dependencies installation (5 min)
3. Conda environment setup (10 min)
4. VERL installation (10 min)
5. Apex compilation (15-20 min)
6. Flash Attention compilation (15-20 min)
7. TransformerEngine compilation (10-15 min)

## Using the Docker Image

### Basic Usage

```bash
# Run interactive container with GPU support
docker run --gpus all -it --rm \
  --name llm-bridge \
  llm-bridge-env:latest

# Run with volume mounts for data persistence
docker run --gpus all -it --rm \
  --name llm-bridge \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  llm-bridge-env:latest
```

### Advanced Usage

```bash
# Run with specific GPUs
docker run --gpus '"device=0,1"' -it --rm \
  --name llm-bridge \
  llm-bridge-env:latest

# Run with shared memory for DataLoader workers
docker run --gpus all -it --rm \
  --name llm-bridge \
  --shm-size=32gb \
  -v $(pwd)/data:/workspace/data \
  llm-bridge-env:latest

# Run with network host mode (for distributed training)
docker run --gpus all -it --rm \
  --name llm-bridge \
  --network host \
  --shm-size=32gb \
  -v $(pwd)/data:/workspace/data \
  llm-bridge-env:latest
```

### Verification

Once inside the container, verify the installation:

```bash
# Run verification script
python /workspace/verify_installation.py

# Expected output:
# ✅ apex.transformer
# ✅ apex.normalization.fused_layer_norm
# ✅ apex.contrib.optimizers.distributed_fused_adam
# ✅ flash_attn
# ✅ verl.trainer
# ✅ ray
# ✅ transformer_engine
# ✅ main_ppo in verl.trainer
# Flash-Attention ver.: 2.6.3 | Ray ver.: 2.48.0 | TransformerEngine ver.: 2.4.0+3cd6870c | Apex ver.: unknown | Torch CUDA: 12.4 | Python: 3.11.x

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
```

## Development Workflow

### 1. Training with Docker

```bash
# Start container with necessary mounts
docker run --gpus all -it --rm \
  --name llm-training \
  --shm-size=32gb \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/training:/workspace/training \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  llm-bridge-env:latest

# Inside container, navigate to training directory
cd /workspace/llm_bridge_prod/train

# Run training (example)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/workspace/data/gsm8k/train.parquet \
    model.partial_pretrain=/workspace/models/Llama-3.2-1B-Instruct
```

### 2. Evaluation with Docker

```bash
# For HLE evaluation
docker run --gpus all -it --rm \
  --name llm-eval-hle \
  -v $(pwd)/evaluation_results:/workspace/evaluation_results \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  llm-bridge-env:latest \
  bash -c "cd /workspace/llm_bridge_prod/eval_hle && python predict.py"

# For DNA evaluation
docker run --gpus all -it --rm \
  --name llm-eval-dna \
  -v $(pwd)/evaluation_results:/workspace/evaluation_results \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  llm-bridge-env:latest \
  bash -c "cd /workspace/llm_bridge_prod/eval_dna && python llm-compe-eval/evaluate_huggingface_models.py --model_name MODEL_NAME"
```

## Troubleshooting

### Common Issues

1. **Out of Memory During Build**
   ```bash
   # Reduce parallel jobs
   docker build --build-arg MAX_JOBS=4 -t llm-bridge-env:latest .
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check host CUDA version
   nvidia-smi
   
   # Ensure driver supports CUDA 12.4
   ```

3. **Build Cache Issues**
   ```bash
   # Clean build without cache
   docker build --no-cache -t llm-bridge-env:latest .
   
   # Remove dangling images
   docker image prune -f
   ```

4. **GPU Not Accessible**
   ```bash
   # Test NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   
   # Check nvidia-container-runtime
   docker info | grep nvidia
   ```

## Docker Compose Alternative

Create `docker-compose.yml` for easier management:

```yaml
version: '3.8'

services:
  llm-bridge:
    image: llm-bridge-env:latest
    build:
      context: .
      dockerfile: Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    volumes:
      - ./data:/workspace/data
      - ./models:/workspace/models
      - ./training:/workspace/training
      - ~/.cache/huggingface:/root/.cache/huggingface
    shm_size: '32gb'
    stdin_open: true
    tty: true
```

Usage:
```bash
# Build and start
docker-compose up -d

# Enter container
docker-compose exec llm-bridge bash

# Stop and remove
docker-compose down
```

## Best Practices

1. **Resource Management**
   - Use `--shm-size` for DataLoader workers
   - Mount HuggingFace cache to avoid re-downloading models
   - Use volume mounts for persistent data

2. **Security**
   - Never hardcode API keys in Dockerfile
   - Use environment variables or Docker secrets
   - Run containers with minimal privileges when possible

3. **Performance**
   - Use host network mode for distributed training
   - Ensure Docker storage driver supports large files
   - Consider using NVMe SSDs for build cache

4. **Maintenance**
   - Tag images with versions: `llm-bridge-env:v1.0`
   - Keep base images updated
   - Regularly prune unused images and containers

## Pushing to Registry

```bash
# Tag for registry
docker tag llm-bridge-env:latest your-registry.com/llm-bridge-env:latest

# Push to registry
docker push your-registry.com/llm-bridge-env:latest

# Pull on another machine
docker pull your-registry.com/llm-bridge-env:latest
```

## Additional Notes

- The image size will be approximately 20-30GB
- First build will download many dependencies
- Subsequent builds will use Docker layer cache
- Consider using a Docker registry for team collaboration
- For production, consider multi-stage builds to reduce final image size