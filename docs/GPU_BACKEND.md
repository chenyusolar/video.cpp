# GPU Backend Support Guide

video.cpp supports multiple GPU backends for accelerated inference. This guide covers installation and configuration of CUDA and Vulkan backends.

## Table of Contents

- [Backend Overview](#backend-overview)
- [CUDA Backend Setup](#cuda-backend-setup)
- [Vulkan Backend Setup](#vulkan-backend-setup)
- [Build with GPU Support](#build-with-gpu-support)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)

---

## Backend Overview

| Backend | Pros | Cons | Recommended For |
|---------|------|------|-----------------|
| **CPU** | Works everywhere, no setup | Slow for large models | Testing, small batches |
| **CUDA** | Fastest on NVIDIA GPUs | NVIDIA GPU only | Production, large models |
| **Vulkan** | Cross-platform, good compatibility | Slightly slower than CUDA | Multi-GPU, portability |

---

## CUDA Backend Setup

### System Requirements

- **GPU**: NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
- **CUDA Toolkit**: 11.8 or later
- **cuDNN**: 8.x (optional, for improved performance)
- **Driver**: Latest NVIDIA driver (525.60+)

### Supported GPUs

| GPU Series | Examples | VRAM | Recommendation |
|------------|----------|------|----------------|
| RTX 30/40 | RTX 3080, 4090 | 10-24GB | ✅ Excellent |
| RTX 20 | RTX 2080, 2080 Ti | 8-11GB | ✅ Good |
| GTX 16 | GTX 1660, 1650 | 6GB | ⚠️ Minimum |
| A100/H100 | A100, H100 | 40-80GB | ✅ Best |

### Installation Steps

#### 1. Install NVIDIA Driver

```bash
# Ubuntu 22.04+
sudo apt update
sudo apt install nvidia-driver-535

# Verify driver
nvidia-smi
```

#### 2. Install CUDA Toolkit

Visit: https://developer.nvidia.com/cuda-downloads

```bash
# Download CUDA 12.x (recommended)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-110
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.0-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

#### 3. Verify CUDA Installation

```bash
nvcc --version
# Should show: release 12.x

export PATH=/usr/local/cuda-12.x/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.x/lib64:$LD_LIBRARY_PATH
```

#### 4. Check GPU Availability

```bash
nvidia-smi
# Shows GPU name, driver version, memory
```

---

## Vulkan Backend Setup

### System Requirements

- **GPU**: Any GPU with Vulkan 1.2+ support (NVIDIA, AMD, Intel, Apple Silicon)
- **Vulkan SDK**: 1.3+ recommended
- **Driver**: Latest GPU driver with Vulkan support

### Supported GPUs

| GPU | Vulkan Support | Performance |
|-----|----------------|-------------|
| NVIDIA RTX 20/30/40 | ✅ Full | Good |
| AMD RX 6000/7000 | ✅ Full | Good |
| Intel Arc | ✅ Full | Good |
| Apple M1/M2/M3 | ✅ Full | Good |
| GTX 10 series | ⚠️ Limited | Medium |

### Installation Steps

#### Windows

1. Visit: https://vulkan.lunarg.com/sdk/home
2. Download Vulkan SDK for Windows
3. Run installer
4. Restart terminal

#### Linux

```bash
# Ubuntu/Debian
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.243.jammy.list https://packages.lunarg.com/1.3.243/jammy/lunarg-vulkan-1.3.243-jammy.list
sudo apt update
sudo apt install vulkan-sdk
```

#### macOS

```bash
# Install via Homebrew
brew install vulkan-validationlayers glslang
```

#### Verify Installation

```bash
vulkaninfo --version
# Should show Vulkan version

# Check GPU
vulkaninfo | grep "GPU"
```

---

## Build with GPU Support

### Quick Setup Script

```bash
# Run environment setup
./scripts/setup.sh --install-all

# Or use PowerShell on Windows
.\scripts\setup.ps1 -InstallAll
```

### Manual Build

#### CUDA Build

```bash
cd core

# Debug build with CUDA
cargo build --features cuda

# Release build with CUDA
cargo build --release --features cuda
```

#### Vulkan Build

```bash
cd core

# Debug build with Vulkan
cargo build --features vulkan

# Release build with Vulkan
cargo build --release --features vulkan
```

### Build Script Usage

```bash
# Using build.ps1 (Windows PowerShell)
.\scripts\build.ps1 -Release -WithCUDA

# Using build.sh (Linux/macOS)
chmod +x scripts/build.sh
./scripts/build.sh -r --with-cuda
```

### Build Options

| Option | Description |
|--------|-------------|
| `--features cuda` | Enable CUDA backend |
| `--features vulkan` | Enable Vulkan backend |
| `--release` | Optimized release build |

---

## Performance Comparison

### Attention Performance (TFLOPS)

| Backend | FP16 | INT8 | INT4 |
|---------|------|------|------|
| CPU (8 cores) | ~0.5 | ~1.0 | ~2.0 |
| CUDA (RTX 4090) | ~330 | ~450 | ~600 |
| Vulkan | ~280 | ~380 | ~500 |

### Memory Requirements (Q4_K_M)

| Resolution | Frames | CUDA VRAM | Vulkan VRAM |
|------------|--------|-----------|-------------|
| 512x512 | 24 | 10GB | 11GB |
| 768x768 | 24 | 18GB | 19GB |
| 512x512 | 49 | 14GB | 15GB |

### Speed Comparison (seconds per frame)

| Backend | 512x512 @ 24f | 768x768 @ 24f |
|---------|---------------|---------------|
| CPU | ~30s | ~90s |
| CUDA | ~0.5s | ~1.5s |
| Vulkan | ~0.7s | ~2.0s |

---

## Troubleshooting

### CUDA Issues

#### "CUDA not found" during build

```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version
```

#### "nvcc: command not found"

```bash
# Install CUDA Toolkit
# Visit: https://developer.nvidia.com/cuda-downloads
```

#### "CUDA out of memory"

```bash
# Reduce batch size or resolution
./bin/video -W 384 -H 384 -f 24 -p "prompt"
```

#### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# If driver is missing, reinstall
sudo apt install nvidia-driver-535
```

### Vulkan Issues

#### "Vulkan not found"

```bash
# Install Vulkan SDK
# Windows: Download from https://vulkan.lunarg.com/
# Linux: sudo apt install vulkan-sdk
```

#### "Vulkan device not found"

```bash
# Check Vulkan devices
vulkaninfo --summary

# Ensure GPU driver supports Vulkan 1.2+
vulkaninfo | grep "Vulkan Version"
```

#### Poor Vulkan performance

```bash
# Use latest GPU drivers
# NVIDIA: sudo apt install nvidia-driver-535
# AMD: sudo apt install mesa-vulkan-drivers
```

### Common Issues

#### Memory allocation failures

```bash
# Set explicit device
export VIDEO_DEVICE_ID=0

# Limit VRAM usage
export VIDEO_VRAM_SIZE_MB=12000
```

#### Slow inference on CPU

```bash
# Use more CPU threads
export VIDEO_NUM_THREADS=16

# Use faster sampler
./bin/video --sampler euler -p "prompt"
```

---

## Environment Variables

### GPU Configuration

```bash
# Backend selection: auto, cpu, cuda, vulkan
export VIDEO_BACKEND=auto

# GPU device ID (for multi-GPU systems)
export VIDEO_DEVICE_ID=0

# VRAM limit (MB)
export VIDEO_VRAM_SIZE_MB=16384

# Memory offload threshold
export VIDEO_OFFLOAD_THRESHOLD_MB=12000
```

### Performance Tuning

```bash
# Enable flash attention (CUDA only)
export VIDEO_USE_FLASH_ATTENTION=true

# Number of CPU threads
export VIDEO_NUM_THREADS=8

# Batch size
export VIDEO_BATCH_SIZE=1
```

---

## Additional Resources

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/)
- [Vulkan SDK Documentation](https://vulkan.lunarg.com/doc/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
