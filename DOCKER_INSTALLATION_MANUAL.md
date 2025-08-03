# Docker Installation Manual for Rocky Linux 9.5

This manual provides step-by-step instructions for installing Docker in the Matsuo Lab LLM Development Competition 2025 environment running on Rocky Linux 9.5.

## Prerequisites

- Rocky Linux 9.5 (Blue Onyx)
- x86_64 architecture
- Root or sudo access
- Active internet connection

## Installation Steps

### 1. Update System Packages

```bash
sudo dnf update -y
```

### 2. Install Required Dependencies

```bash
sudo dnf install -y dnf-plugins-core
```

### 3. Add Docker Repository

```bash
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
```

### 4. Install Docker Engine

```bash
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 5. Start and Enable Docker Service

```bash
# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Verify Docker service status
sudo systemctl status docker
```

### 6. Configure User Permissions (Optional)

To run Docker commands without sudo:

```bash
# Add current user to docker group
sudo usermod -aG docker $USER

# Apply group changes (logout/login or use newgrp)
newgrp docker
```

### 7. Verify Installation

```bash
# Check Docker version
docker --version

# Run test container
docker run hello-world
```

## Post-Installation Configuration

### Configure Docker for HPC Environment

Since this is an HPC environment with specific module requirements, consider these configurations:

1. **Configure Docker storage driver** (if needed):
```bash
# Edit Docker daemon configuration
sudo nano /etc/docker/daemon.json
```

Add configuration (example):
```json
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ]
}
```

2. **Restart Docker after configuration changes**:
```bash
sudo systemctl restart docker
```

### Configure Docker for GPU Support (NVIDIA Container Toolkit)

If you need GPU support for your containers:

1. **Add NVIDIA repository**:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/rhel9.5/libnvidia-container.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

2. **Install NVIDIA Container Toolkit**:
```bash
sudo dnf clean expire-cache
sudo dnf install -y nvidia-container-toolkit
```

3. **Configure Docker to use NVIDIA runtime**:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. **Test GPU access**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Environment-Specific Considerations

### Working with HPC Modules

When using Docker in this environment, you may need to consider interactions with existing modules:

```bash
# Load required modules before Docker operations
module reset
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load miniconda/24.7.1-py311
```

### Docker Images for LLM Development

Create a Dockerfile for your LLM development environment:

```dockerfile
FROM nvidia/cuda:12.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other ML libraries
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project-specific dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /workspace
```

## Troubleshooting

### Common Issues and Solutions

1. **Permission Denied Error**:
   ```bash
   # Ensure Docker service is running
   sudo systemctl status docker
   
   # Check user is in docker group
   groups $USER
   ```

2. **Repository Not Found**:
   ```bash
   # For Rocky Linux 9, use CentOS 9 repository
   sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
   ```

3. **Conflict with Podman**:
   ```bash
   # Rocky Linux may have Podman pre-installed
   # Remove Podman if conflicts occur
   sudo dnf remove podman buildah
   ```

4. **SELinux Issues**:
   ```bash
   # Check SELinux status
   getenforce
   
   # If needed, configure SELinux for Docker
   sudo setsebool -P container_manage_cgroup on
   ```

## Security Considerations

1. **Never run Docker daemon as root in production**
2. **Use Docker secrets for sensitive data**
3. **Implement resource limits for containers**:
   ```bash
   docker run --memory="2g" --cpus="2" your-image
   ```

4. **Regular security updates**:
   ```bash
   sudo dnf update docker-ce docker-ce-cli containerd.io
   ```

## Useful Docker Commands for LLM Development

```bash
# Build custom image
docker build -t llm-dev:latest .

# Run interactive container with GPU
docker run -it --gpus all -v $HOME/data:/data llm-dev:latest bash

# Run training with mounted volumes
docker run --gpus all \
  -v $HOME/model:/model \
  -v $HOME/data:/data \
  -v $HOME/training:/workspace \
  llm-dev:latest python train.py

# Check resource usage
docker stats

# Clean up unused resources
docker system prune -a
```

## Integration with SLURM

For HPC environments using SLURM, you can run Docker containers within SLURM jobs:

```bash
#!/bin/bash
#SBATCH --job-name=docker-llm
#SBATCH --partition=YOUR_TEAM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --time=24:00:00

# Load modules
module load docker

# Run Docker container
docker run --gpus all -v $HOME:/home llm-dev:latest python train.py
```

## References

- [Docker Official Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Rocky Linux Documentation](https://docs.rockylinux.org/)