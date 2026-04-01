# video.cpp - Technical Specification

## Overview

video.cpp is a local video generation inference engine, inspired by llama.cpp's approach to LLM inference. It provides a hardware-accelerated, dependency-free runtime for video generation models like LTX-2.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / API                           │
│                    (C++ CLI, Go SDK)                       │
├─────────────────────────────────────────────────────────────┤
│                       FFI Bridge                           │
│               (C ABI, Go/Rust FFI bindings)                │
├─────────────────────────────────────────────────────────────┤
│                      Core Runtime (Rust)                   │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Scheduler   │ │   Pipeline    │ │   Memory Manager     │  │
│  │ (DDIM/Euler│ │   Engine      │ │                     │  │
│  │ /DPM++)    │ │               │ │                     │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                       Model Layer                          │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ Transformer │ │     VAE       │ │   Text Encoder      │  │
│  │   (DiT)     │ │  (Enc/Dec)    │ │     (CLIP/T5)       │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Backend Abstraction                    │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │    CUDA     │ │    Vulkan     │ │        CPU          │  │
│  │  (cuBLAS)   │ │   (GLSL)      │ │      (BLAS)         │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Model Format (GGUF-VID)

GGUF-VID (.gguv) is based on the GGUF format by llama.cpp, extended for video generation.

### File Structure

```
┌─────────────────────────────────────────────────────────────┐
│ GGUF-VID Header (16 bytes)                                  │
│   - Magic: "GGUV" (4 bytes)                                │
│   - Version: u32                                           │
│   - Tensor Count: u64                                      │
├─────────────────────────────────────────────────────────────┤
│ Metadata (256 bytes reserved)                              │
│   - model_type: string                                     │
│   - latent_shape: [T, H, W, C]                            │
│   - latent_channels: u32                                  │
│   - fps: u32                                              │
│   - has_audio: bool                                       │
│   - vae_encoder/decoder: bool                             │
│   - text_encoder: bool                                    │
│   - quantization_type: string                            │
├─────────────────────────────────────────────────────────────┤
│ Tensor Info Table                                         │
│   - For each tensor:                                      │
│     * name: string                                        │
│     * n_dims: u32                                         │
│     * shape: [u64; n_dims]                                │
│     * dtype: u32                                          │
│     * offset: u64                                         │
├─────────────────────────────────────────────────────────────┤
│ Tensor Data (mmap-able)                                   │
│   - Raw tensor data, quantized if applicable              │
└─────────────────────────────────────────────────────────────┘
```

### Tensor Naming Convention

| Pattern | Description | Shape |
|---------|-------------|-------|
| `vae.encoder.*` | VAE Encoder weights | varies |
| `vae.decoder.*` | VAE Decoder weights | varies |
| `transformer.*` | DiT/UNet weights | [B, T, H, W, C] |
| `text_encoder.*` | CLIP/T5 weights | varies |
| `temporal_attn.*` | Time attention weights | varies |

### Data Types

| ID | Type | Description |
|----|------|-------------|
| 0 | F32 | 32-bit float |
| 1 | F16 | 16-bit float (half precision) |
| 2 | BF16 | Brain float 16 |
| 3 | INT8 | 8-bit integer |
| 4 | INT4 | 4-bit integer (NF4) |
| 5 | Q8_0 | LLAMA-style Q8_0 quantization |

## Core Traits

### Backend Trait

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn tensor_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_conv2d(&self, input: &Tensor, weight: &Tensor, opts: Conv2dOpts) -> Result<Tensor>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor>;
    fn temporal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, t: u32) -> Result<Tensor>;
    fn alloc_tensor(&self, shape: TensorShape, dtype: DType) -> Result<Tensor>;
    fn copy_to_device(&self, data: &[f32], tensor: &mut Tensor) -> Result<()>;
    fn copy_to_host(&self, tensor: &Tensor) -> Result<Vec<f32>>;
    fn randn(&self, shape: TensorShape) -> Result<Tensor>;
    fn randn_like(&self, tensor: &Tensor) -> Result<Tensor>;
    fn synchronize(&self) -> Result<()>;
    fn memory_allocated(&self) -> u64;
    fn memory_reserved(&self) -> u64;
}
```

### Scheduler Trait

```rust
pub trait Scheduler: Send + Sync {
    fn step(&mut self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor>;
    fn add_noise(&mut self, latent: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor>;
    fn timesteps(&self) -> Vec<f32>;
    fn set_timesteps(&mut self, num_steps: u32);
}
```

### Supported Schedulers

| Name | Description | Speed | Quality |
|------|-------------|-------|---------|
| `euler` | Euler method | Fast | Good |
| `euler_a` | Euler ancestor | Medium | Better |
| `ddim` | DDIM | Medium | Very Good |
| `dpm++` | DPM++ 2M | Slow | Best |

## Pipeline Flow

```
1. Text Encoding
   prompt → CLIP/T5 → context embeddings

2. Latent Initialization
   noise = randn([B, T, H/8, W/8, C_latent])

3. Diffusion Loop (for each timestep)
   a. Transformer forward pass
      noise → transformer → noise_pred
   b. CFG application (if enabled)
      noise_pred = cfg_scale * (cond - uncond) + uncond
   c. Scheduler step
      noise = scheduler.step(noise, timestep, noise_pred)

4. VAE Decode
   latent → VAE.decoder → frames [B, T, C, H, W]

5. Video Encoding
   frames → encoder → .mp4/.webm
```

## Memory Requirements

| Resolution | Frames | Latent Shape | VRAM (FP16) |
|------------|--------|--------------|-------------|
| 512x512 | 24 | 1x16x64x64x16 | ~12GB |
| 768x768 | 24 | 1x16x96x96x16 | ~18GB |
| 1024x1024 | 16 | 1x8x128x128x16 | ~16GB |

## Configuration

### Environment Variables

```env
VIDEO_MODEL_PATH=models/ltx2.gguv
VIDEO_BACKEND=auto           # auto, cpu, cuda, vulkan
VIDEO_DEVICE_ID=0
VIDEO_USE_GPU=true
VIDEO_VRAM_SIZE_MB=24576
VIDEO_AVAILABLE_MEMORY_MB=32768
VIDEO_OFFLOAD_THRESHOLD_MB=16000
VIDEO_USE_OFFLOAD=false
VIDEO_STEPS=30
VIDEO_GUIDANCE_SCALE=7.5
VIDEO_SAMPLER=euler          # euler, ddim, dpm++
VIDEO_FPS=24
VIDEO_QUANTIZATION=auto      # auto, none, int8, nf4, fp8
VIDEO_QUANT_BLOCK_SIZE=32
VIDEO_NUM_THREADS=8
VIDEO_USE_FLASH_ATTENTION=true
```

## API

### C API

```c
#include <video.h>

model_handle model;
video_load("model.gguv", &model);

generate_request req = {
    .prompt = "a dragon flying",
    .frames = 24,
    .width = 512,
    .height = 512,
    .steps = 30,
    .sampler = "euler",
    .cfg_scale = 7.5,
    .seed = -1,
};

video_output output;
video_generate(model, req, &output);

video_free(model);
```

### Go SDK

```go
client, _ := video.Load("model.gguv")
result, _ := client.Generate(video.GenerateRequest{
    Prompt:  "a dragon flying",
    Frames:  24,
    Width:   512,
    Height:  512,
    Steps:   30,
    Sampler: "euler",
})
result.Save("output.mp4")
```

## Performance Targets

| Operation | CPU (FP32) | CUDA (FP16) | Vulkan (FP16) |
|-----------|------------|-------------|---------------|
| Latent init (512x512, 24f) | <1s | <0.5s | <0.5s |
| Single diffusion step | ~30s | ~2s | ~3s |
| Full generation (30 steps) | ~15min | ~60s | ~90s |
| VAE decode | ~10s | ~2s | ~3s |
| Total 24-frame video | ~20min | ~65s | ~100s |

## TODO

- [x] Core Rust runtime
- [x] GGUF-VID loader
- [x] CPU backend
- [x] Scheduler implementations (Euler, DDIM, DPM++)
- [ ] CUDA backend with Flash Attention
- [ ] Vulkan backend
- [ ] Quantization (int8, nf4)
- [ ] Model converter (PyTorch → GGUF-VID)
- [ ] API server
- [ ] Web UI

## References

- [LTX-2 Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Format](https://github.com/ggerganov/gguf)
