# video.cpp

**本地视频生成推理引擎** - 视频 AI 领域的 llama.cpp

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)

## 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [完整安装指南](#完整安装指南)
- [模型下载](#模型下载)
- [编译项目](#编译项目)
- [使用示例](#使用示例)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 特性

- **GGUF 模型支持** - 支持 llama.cpp 风格的量化模型 (LTX-2.3, etc.)
- **多后端支持** - CUDA, Vulkan, CPU 计算后端
- **Rust 核心** - 高性能运行时
- **多任务管道** - Text-to-Video, Image-to-Video, Video-to-Video
- **视频编码** - H.264, H.265, VP9 通过 FFmpeg
- **Go SDK** - 轻松集成到 Go 应用
- **C API** - FFI 绑定 C/C++, Python 等语言
- **CLI 工具** - 简单命令行界面

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/video-ai/video.cpp.git
cd video.cpp
```

### 2. 安装依赖

**Rust (必需)**
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 或使用包管理器
# Ubuntu/Debian
sudo apt install rustc cargo

# macOS
brew install rust
```

**Go (必需，用于 CLI)**
```bash
# 安装 Go 1.21+
# https://go.dev/dl/

# 或使用包管理器
# Ubuntu
sudo apt install golang-go

# macOS
brew install go
```

**CMake (必需)**
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake
```

**FFmpeg (可选，用于视频编码)**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载 https://ffmpeg.org/download.html
```

**Python (可选，用于模型下载)**
```bash
# 需要 Python 3.8+
python --version
```

### 3. 下载模型

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载 LTX-2.3-GGUF 模型 (约 15GB)
python scripts/download_model.py --quant Q4_K_M
```

### 4. 编译

```bash
# 编译 Rust 核心
cd core
cargo build --release
cd ..

# 编译 CLI
cd cli
go build -o ../bin/video .
cd ..
```

### 5. 运行

```bash
# 文生视频
./bin/video -m models/ltx-2.3-22b-dev-Q4_K_M.gguf -p "a dragon flying over city" -f 24 -o dragon.mp4

# 查看帮助
./bin/video --help
```

---

## 完整安装指南

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 内存 | 16GB | 32GB+ |
| 显存 | 8GB (Q4_K_S) | 16GB+ (Q4_K_M) |
| 磁盘 | 30GB | 50GB+ |
| OS | Linux/macOS/Windows | Linux (Ubuntu 22.04+) |

### Linux (Ubuntu 22.04)

```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    libssl-dev \
    pkg-config \
    ffmpeg \
    python3 \
    python3-pip

# 2. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 3. 安装 Go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# 4. 验证安装
rustc --version
go version
cmake --version
ffmpeg -version
```

### macOS

```bash
# 1. 安装 Homebrew (如果还没有)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装依赖
brew install rust go cmake ffmpeg

# 3. 验证安装
rustc --version
go version
cmake --version
ffmpeg -version
```

### Windows

```bash
# 1. 安装 Rust (https://rustup.rs)
# 2. 安装 Go (https://go.dev/dl/)
# 3. 安装 CMake (https://cmake.org/download/)
# 4. 安装 MSVC Build Tools
# 5. 安装 FFmpeg (https://ffmpeg.org/download.html#build-windows)
```

---

## 模型下载

### LTX-2.3 模型系列

LTX-2.3 是 Lightricks 开发的高质量文生视频模型，支持视频和音频联合生成。

#### 下载脚本

```bash
python scripts/download_model.py --help
```

**选项说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 模型保存目录 | `./models` |
| `--quant` | 量化版本 | `Q4_K_M` |
| `--include-text-encoder` | 下载文本编码器 | `true` |
| `--include-vae` | 下载 VAE 模型 | `true` |

#### 量化版本对比

| 版本 | 文件名 | 大小 | 质量 | VRAM | 推荐 |
|------|--------|------|------|------|------|
| Q4_K_S | `ltx-2.3-22b-dev-Q4_K_S.gguf` | 13.7GB | 良好 | 8GB | 小显存 |
| **Q4_K_M** | `ltx-2.3-22b-dev-Q4_K_M.gguf` | 15.1GB | 均衡 | 12GB | ✅推荐 |
| Q5_K_S | `ltx-2.3-22b-dev-Q5_K_S.gguf` | 15.8GB | 更好 | 14GB | 高质量 |
| Q8_0 | `ltx-2.3-22b-dev-Q8_0.gguf` | 22.8GB | 最好 | 20GB | 最高质量 |

#### 推荐下载 (Q4_K_M)

```bash
# 推荐配置 - 平衡质量和速度
python scripts/download_model.py --quant Q4_K_M
```

#### 完整下载命令

```bash
# 创建模型目录
mkdir -p models
cd models

# 下载 DiT 主模型
python ../scripts/download_model.py --quant Q4_K_M

# 下载后手动获取 VAE 和文本编码器 (如果需要)
# 使用 huggingface-cli 或 Python 脚本
```

#### 手动下载

如果你无法使用脚本，可以手动从 Hugging Face 下载：

**主模型 (DiT)**
```
https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main
# 下载: ltx-2.3-22b-dev-Q4_K_M.gguf (约 15GB)
```

**VAE 模型**
```
https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/vae
# 下载:
#   - ltx-2.3-22b-dev_video_vae.safetensors
#   - ltx-2.3-22b-dev_audio_vae.safetensors
```

**文本编码器**
```
https://huggingface.co/unsloth/LTX-2.3-GGUF/tree/main/text_encoders
# 或
https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF/tree/main
# 下载:
#   - gemma-3-12b-it-qat-Q4_K_M.gguf
#   - mmproj-BF16.gguf
#   - ltx-2.3-22b-dev_embeddings_connectors.safetensors
```

#### 手动使用 huggingface_hub

```python
from huggingface_hub import hf_hub_download

# 下载 DiT 模型
dit_model = hf_hub_download(
    repo_id="unsloth/LTX-2.3-GGUF",
    filename="ltx-2.3-22b-dev-Q4_K_M.gguf"
)

# 下载 VAE
video_vae = hf_hub_download(
    repo_id="unsloth/LTX-2.3-GGUF",
    filename="vae/ltx-2.3-22b-dev_video_vae.safetensors"
)

# 下载 Gemma 文本编码器
gemma_model = hf_hub_download(
    repo_id="unsloth/gemma-3-12b-it-qat-GGUF",
    filename="gemma-3-12b-it-qat-Q4_K_M.gguf"
)
```

---

## 编译项目

### 编译 Rust 核心

```bash
cd video.cpp/core

# Debug 模式 (快速编译)
cargo build

# Release 模式 (优化编译)
cargo build --release

# 带 CUDA 支持 (需要 CUDA Toolkit)
cargo build --release --features cuda
```

编译产物位于 `core/target/release/`：
- `libvideo_core.rlib` - Rust 库
- `libvideo_core.so` (Linux) / `libvideo_core.dylib` (macOS) / `video_core.dll` (Windows)
- `video` - 可执行 CLI

### 编译 CLI

```bash
cd video.cpp/cli

# 编译 Go CLI
go build -o ../bin/video .

# Windows
go build -o ../bin/video.exe .
```

### 编译 Go SDK

```bash
cd video.cpp/go

# 编译为库
go build -o video.a .

# 或直接使用 (需要 Rust 核心库)
go build ./...
```

### 一键编译脚本

```bash
# Linux/macOS
chmod +x scripts/build.sh
./scripts/build.sh

# Windows PowerShell
./scripts/build.ps1
```

---

## 使用示例

### CLI 文生视频

```bash
# 基础用法
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a dragon flying over city" \
  -f 24 \
  -o dragon.mp4

# 高级设置
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "cyberpunk street at night with neon lights" \
  --steps 50 \
  --cfg 8.0 \
  --sampler euler \
  --fps 30 \
  -W 768 -H 768 \
  -f 24 \
  -o cyberpunk.mp4

# 使用随机种子 (可复现)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a waterfall in the jungle" \
  --seed 42 \
  -f 24 \
  -o waterfall.mp4

# 详细输出
./bin/video -v \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a spaceship launching" \
  -f 24
```

### Go SDK

```go
package main

import (
    "fmt"
    "log"

    video "github.com/video-ai/video.cpp/go"
)

func main() {
    // 加载模型
    client, err := video.Load("models/ltx-2.3-22b-dev-Q4_K_M.gguf")
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // 设置生成参数
    req := video.GenerateRequest{
        Prompt:   "a dragon flying over city",
        NegativePrompt: "blurry, low quality",
        Frames:   24,
        Width:    512,
        Height:   512,
        FPS:      24,
        Steps:    30,
        Sampler:  "euler",
        CFGScale: 7.5,
        Seed:     42,
    }

    // 进度回调
    callback := func(progress, total int) {
        fmt.Printf("\rProgress: %d/%d (%d%%)", progress, total, progress*100/total)
    }

    // 生成视频
    result, err := client.GenerateWithCallback(req, callback)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("\nGeneration time: %dms\n", result.GenerationTimeMs)

    // 保存视频
    if err := result.Save("output.mp4"); err != nil {
        log.Fatal(err)
    }
}
```

### C API

```c
#include "include/video.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    model_handle handle;
    
    // 加载模型
    if (video_load("models/ltx-2.3-22b-dev-Q4_K_M.gguf", &handle) != VIDEO_OK) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // 设置生成参数
    generate_request req = {
        .prompt = "a dragon flying over city",
        .negative_prompt = "blurry, low quality",
        .frames = 24,
        .width = 512,
        .height = 512,
        .fps = 24,
        .steps = 30,
        .sampler = "euler",
        .cfg_scale = 7.5,
        .seed = 42,
    };

    // 生成视频
    video_output output;
    if (video_generate(handle, req, &output) != VIDEO_OK) {
        fprintf(stderr, "Generation failed\n");
        video_free(handle);
        return 1;
    }

    // 保存视频
    FILE *f = fopen("output.mp4", "wb");
    fwrite(output.data, 1, output.size, f);
    fclose(f);

    // 释放内存
    free(output.data);
    video_free(handle);

    printf("Video saved! Size: %zux%zu, %d bytes\n",
           (size_t)output.width, (size_t)output.height, (size_t)output.size);

    return 0;
}
```

编译:
```bash
gcc -o video_example video_example.c \
    -L core/target/release -lvideo_core \
    -lm -lpthread
```

---

## 配置说明

### 环境变量

在 `.env` 文件中设置，或导出环境变量：

```bash
# 模型路径
export VIDEO_MODEL_PATH=models/ltx-2.3-22b-dev-Q4_K_M.gguf
export VIDEO_VAE_PATH=models/vae/ltx-2.3-22b-dev_video_vae.safetensors
export VIDEO_AUDIO_VAE_PATH=models/vae/ltx-2.3-22b-dev_audio_vae.safetensors
export VIDEO_TEXT_ENCODER_PATH=models/text_encoders/gemma-3-12b-it-qat-Q4_K_M.gguf

# 后端选择: auto, cpu, cuda, vulkan
export VIDEO_BACKEND=auto

# GPU 设置
export VIDEO_DEVICE_ID=0
export VIDEO_USE_GPU=true
export VIDEO_VRAM_SIZE_MB=16384

# 内存管理
export VIDEO_AVAILABLE_MEMORY_MB=32768
export VIDEO_OFFLOAD_THRESHOLD_MB=16000

# 生成参数
export VIDEO_STEPS=30
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=euler
export VIDEO_FPS=24

# 量化
export VIDEO_QUANTIZATION=Q4_K_M

# 性能
export VIDEO_NUM_THREADS=8
export VIDEO_USE_FLASH_ATTENTION=true

# 日志
export RUST_LOG=info
```

### .env 文件示例

创建 `video.cpp/.env`:

```env
# ============ 模型配置 ============
VIDEO_MODEL_PATH=models/ltx-2.3-22b-dev-Q4_K_M.gguf
VIDEO_VAE_PATH=models/vae/ltx-2.3-22b-dev_video_vae.safetensors
VIDEO_AUDIO_VAE_PATH=models/vae/ltx-2.3-22b-dev_audio_vae.safetensors
VIDEO_TEXT_ENCODER_PATH=models/text_encoders/gemma-3-12b-it-qat-Q4_K_M.gguf
VIDEO_TEXT_ENCODER_MMPROJ=models/text_encoders/mmproj-BF16.gguf

# ============ 后端配置 ============
VIDEO_BACKEND=auto
VIDEO_DEVICE_ID=0

# ============ GPU 配置 ============
VIDEO_USE_GPU=true
VIDEO_VRAM_SIZE_MB=16384
VIDEO_AVAILABLE_MEMORY_MB=32768
VIDEO_OFFLOAD_THRESHOLD_MB=12000

# ============ 生成参数 ============
VIDEO_STEPS=30
VIDEO_GUIDANCE_SCALE=7.5
VIDEO_SAMPLER=euler
VIDEO_FPS=24

# ============ 量化配置 ============
VIDEO_QUANTIZATION=Q4_K_M
VIDEO_QUANT_BLOCK_SIZE=32

# ============ 性能优化 ============
VIDEO_NUM_THREADS=8
VIDEO_USE_FLASH_ATTENTION=true

# ============ 日志 ============
RUST_LOG=info
```

### CLI 参数覆盖

CLI 参数会覆盖环境变量设置：

```bash
# 环境变量 VIDEO_STEPS=30
# CLI 指定 --steps 50
# 最终使用 50
./bin/video --steps 50 -p "prompt"
```

---

## API 参考

### CLI 参数

| 参数 | 短 | 类型 | 默认值 | 说明 |
|------|----|------|--------|------|
| `--model` | `-m` | string | 必需 | 模型文件路径 |
| `--prompt` | `-p` | string | 必需 | 文本提示 |
| `--negative-prompt` | `-n` | string | "" | 负面提示 |
| `--output` | `-o` | string | output.mp4 | 输出路径 |
| `--frames` | `-f` | int | 24 | 帧数 (需能被 9 整除) |
| `--width` | `-W` | int | 512 | 宽度 (需能被 32 整除) |
| `--height` | `-H` | int | 512 | 高度 (需能被 32 整除) |
| `--fps` | | int | 24 | 帧率 |
| `--steps` | | int | 30 | 扩散步数 |
| `--sampler` | | string | euler | 采样器: euler, ddim, dpm++ |
| `--cfg` | | float | 7.5 | CFG 引导强度 |
| `--seed` | | int64 | -1 | 随机种子 (-1 随机) |
| `--backend` | | string | cpu | 后端: cpu, cuda, vulkan |
| `--verbose` | `-v` | flag | false | 详细输出 |

### 采样器说明

| 采样器 | 速度 | 质量 | 适用场景 |
|--------|------|------|----------|
| `euler` | 最快 | 良好 | 快速预览 |
| `euler_a` | 快 | 更好 | 平衡质量速度 |
| `ddim` | 中 | 好 | 稳定生成 |
| `dpm++` | 慢 | 最好 | 高质量最终输出 |

### Go SDK 类型

```go
type GenerateRequest struct {
    Prompt         string   // 文本提示
    NegativePrompt string   // 负面提示
    Frames         int      // 帧数
    Width          int      // 宽度
    Height         int      // 高度
    FPS            int      // 帧率
    Steps          int      // 步数
    Sampler        string   // 采样器
    CFGScale       float32  // CFG 强度
    Seed           int64    // 种子
    Backend        string   // 后端
}

type VideoOutput struct {
    Data             []byte  // 视频数据
    Width            int     // 宽度
    Height           int     // 高度
    FPS              int     // 帧率
    GenerationTimeMs int64   // 生成耗时(ms)
}
```

### C API

```c
// 加载模型
video_error video_load(const char* model_path, model_handle* out_handle);

// 释放模型
void video_free(model_handle handle);

// 文生视频
video_error video_generate(
    model_handle handle,
    generate_request req,
    video_output* out
);

// 图生视频
video_error video_generate_image_to_video(
    model_handle handle,
    const uint8_t* init_image,
    size_t image_size,
    const char* prompt,
    int32_t width,
    int32_t height,
    int32_t frames,
    float strength,
    int32_t steps,
    float cfg_scale,
    int64_t seed,
    video_output* out
);

// 视频转视频
video_error video_generate_video_to_video(
    model_handle handle,
    const uint8_t* init_video,
    size_t video_size,
    const char* prompt,
    int32_t width,
    int32_t height,
    int32_t frames,
    float strength,
    int32_t steps,
    float cfg_scale,
    int64_t seed,
    video_output* out
);

// 设置后端
video_error video_set_backend(video_backend backend);

// 获取版本
const char* video_get_version();
```

---

## 项目结构

```
video.cpp/
├── core/                           # Rust 核心库
│   ├── Cargo.toml                 # Rust 项目配置
│   └── src/
│       ├── lib.rs               # 库入口
│       ├── bin/main.rs          # CLI 可执行文件入口
│       ├── config.rs            # 配置管理
│       ├── libcore/             # 核心抽象
│       │   ├── tensor.rs        # 张量定义
│       │   ├── traits.rs        # 核心 trait
│       │   └── context.rs       # 上下文
│       ├── model/               # 模型层
│       │   ├── gguf.rs         # GGUF 格式解析
│       │   ├── dit.rs          # DiT transformer
│       │   ├── vae.rs          # VAE 编解码
│       │   ├── text_encoder.rs # 文本编码器
│       │   └── loader.rs        # 模型加载
│       ├── scheduler/           # 扩散调度器
│       │   ├── euler.rs        # Euler
│       │   ├── ddim.rs         # DDIM
│       │   └── dpmpp.rs        # DPM++
│       ├── pipeline/            # 推理管道
│       ├── backend/             # 计算后端
│       │   ├── cpu.rs          # CPU 后端
│       │   ├── cuda/           # CUDA 后端
│       │   └── vulkan/         # Vulkan 后端
│       ├── encoder/            # 视频编码
│       └── ffi/                # FFI 绑定
│
├── cli/                           # Go CLI
│   └── main.go
│
├── go/                            # Go SDK
│   ├── client.go
│   ├── internal/
│   │   └── bindings.go
│   └── examples/
│       └── simple/
│
├── include/                       # C 头文件
│   └── video.h
│
├── scripts/                        # 工具脚本
│   ├── build.sh                  # 编译脚本
│   └── download_model.py         # 模型下载
│
├── model/                         # 模型配置
│   └── config.json
│
├── tests/                         # 测试
│
├── .env.example                  # 环境变量示例
├── CMakeLists.txt                # CMake 配置
├── README.md                     # 本文档
└── SPEC.md                       # 技术规范
```

---

## 常见问题

### Q: 显存不足怎么办？

A: 选择更小的量化版本或降低分辨率：
```bash
# 使用 Q4_K_S (8GB VRAM)
python scripts/download_model.py --quant Q4_K_S

# 或降低分辨率
./bin/video -W 384 -H 384 -f 24 -p "prompt"
```

### Q: 生成速度太慢？

A: 尝试以下优化：
1. 使用更少的步数：`--steps 20`
2. 使用更快的采样器：`--sampler euler`
3. 使用 CUDA 后端：`--backend cuda`
4. 降低分辨率：`--W 384 --H 384`

### Q: 模型下载失败？

A: 检查网络或使用代理：
```bash
# 设置代理
export HTTPS_PROXY=http://127.0.0.1:7890

# 或使用 huggingface_hub
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('unsloth/LTX-2.3-GGUF', 'ltx-2.3-22b-dev-Q4_K_M.gguf'))"
```

### Q: FFmpeg 未找到？

A: 安装 FFmpeg 或视频将被保存为原始 RGB 数据：
```bash
# Ubuntu
sudo apt install ffmpeg

# 视频会保存为 .rgb 格式，需要手动转换
ffmpeg -f rawvideo -pixel_format rgb24 -video_size 512x512 -framerate 24 -i output.rgb output.mp4
```

### Q: 如何复现相同结果？

A: 使用固定种子：
```bash
./bin/video --seed 42 -p "prompt" -f 24 -o output.mp4
```

### Q: 支持哪些模型？

A: 目前主要支持：
- LTX-2.3 (22B) - ✅ 完全支持
- LTX-2 (14B) - 🔜 计划中

---

## 许可证

Apache 2.0 / MIT

## 参考链接

- [LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) - 模型权重
- [LTX-2 论文](https://arxiv.org/abs/2601.03233) - 研究论文
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 架构灵感
- [LTX-Video 文档](https://docs.ltx.video/) - 官方文档

---

**star 并支持我们的项目！** ⭐
