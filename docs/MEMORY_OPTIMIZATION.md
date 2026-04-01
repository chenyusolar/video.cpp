# Video Memory Optimization Guide

针对不同显存大小的显卡，提供优化的配置方案和运行实例。

## 显存需求概览

| 显存 | 量化版本 | 推荐分辨率 | 推荐帧数 | 最大步数 | 采样器 |
|------|----------|------------|----------|----------|--------|
| **8GB** | Q4_K_S | 256-384 | 9-16 | 15-20 | euler |
| **12GB** | Q4_K_M | 384-512 | 16-24 | 20-25 | euler |
| **16GB** | Q4_K_M | 512-640 | 24-32 | 25-30 | euler/euler_a |
| **24GB** | Q5_K_S | 640-768 | 32-48 | 30-40 | euler_a |
| **32GB** | Q5_K_M | 768-1024 | 48-64 | 40-50 | ddim/dpm++ |
| **48GB** | Q8_0 | 1024-1280 | 64-96 | 50-60 | ddim |
| **96GB** | BF16/FP16 | 1280+ | 96+ | 50-80 | dpm++ |

---

## 8GB 显存配置

### 特点
- 显存非常有限
- 需要使用最小量化版本
- 分辨率和帧数受限
- 建议作为开发测试使用

### 推荐配置

```bash
# 使用 Q4_K_S 量化版本 (约 13.7GB)
# 需要模型: ltx-2.3-22b-dev-Q4_K_S.gguf

# 最低配置 (256x256)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_S.gguf \
  -p "a cat" \
  -W 256 -H 256 \
  -f 9 \
  --steps 15 \
  --sampler euler \
  -o output.mp4

# 标准配置 (384x384)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_S.gguf \
  -p "a dragon flying" \
  -W 384 -H 384 \
  -f 16 \
  --steps 20 \
  --sampler euler \
  --cfg 6.0 \
  -o output.mp4
```

### 环境变量

```bash
# 8GB 显存优化
export VIDEO_BACKEND=cpu
export VIDEO_VRAM_SIZE_MB=8192
export VIDEO_AVAILABLE_MEMORY_MB=8192
export VIDEO_STEPS=20
export VIDEO_SAMPLER=euler
export VIDEO_OFFLOAD_THRESHOLD_MB=6000

# 降低线程数
export VIDEO_NUM_THREADS=4
```

### 性能预期
- 首次生成: 约 5-10 分钟
- 后续生成: 约 3-5 分钟
- 内存占用: ~7.5GB

---

## 12GB 显存配置

### 特点
- 可以使用 Q4_K_M 量化
- 支持中等分辨率
- 帧数有限

### 推荐配置

```bash
# 使用 Q4_K_M 量化版本 (约 15.1GB)

# 标准配置 (512x512, 24帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a dragon flying over city" \
  -W 512 -H 512 \
  -f 24 \
  --steps 25 \
  --sampler euler \
  -o dragon.mp4

# 质量配置 (640x360, 32帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "cyberpunk street at night" \
  -W 640 -H 360 \
  -f 32 \
  --steps 30 \
  --sampler euler_a \
  --cfg 7.5 \
  -o cyberpunk.mp4
```

### 环境变量

```bash
# 12GB 显存优化
export VIDEO_BACKEND=cpu
export VIDEO_VRAM_SIZE_MB=12288
export VIDEO_AVAILABLE_MEMORY_MB=12288
export VIDEO_STEPS=25
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=euler
export VIDEO_NUM_THREADS=6
```

### 性能预期
- 首次生成: 约 3-5 分钟
- 后续生成: 约 2-3 分钟
- 内存占用: ~11GB

---

## 16GB 显存配置

### 特点
- 流畅运行 Q4_K_M
- 可支持较高分辨率
- 帧数适中

### 推荐配置

```bash
# 使用 Q4_K_M 量化版本

# 高质量配置 (640x640, 32帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a waterfall in jungle with sunlight" \
  -W 640 -H 640 \
  -f 32 \
  --steps 30 \
  --sampler euler_a \
  --cfg 7.5 \
  -o waterfall.mp4

# 电影配置 (768x512, 48帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a spaceship launching into space" \
  -W 768 -H 512 \
  -f 48 \
  --steps 35 \
  --sampler euler \
  --cfg 8.0 \
  --fps 30 \
  -o spaceship.mp4

# 使用 CUDA 加速 (如果有)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  -p "a dragon flying over mountains" \
  -W 512 -H 512 \
  -f 24 \
  --backend cuda \
  --steps 30 \
  -o dragon_cuda.mp4
```

### 环境变量

```bash
# 16GB 显存优化
export VIDEO_BACKEND=auto
export VIDEO_VRAM_SIZE_MB=16384
export VIDEO_AVAILABLE_MEMORY_MB=16384
export VIDEO_OFFLOAD_THRESHOLD_MB=12000
export VIDEO_STEPS=30
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=euler_a
export VIDEO_NUM_THREADS=8
export VIDEO_USE_FLASH_ATTENTION=true
```

### 性能预期
- CPU: 约 2-3 分钟
- CUDA: 约 30-60 秒
- 内存占用: ~14GB

---

## 24GB 显存配置

### 特点
- 可以使用 Q5_K_S 量化
- 支持高分辨率
- 帧数充足

### 推荐配置

```bash
# 使用 Q5_K_S 量化版本 (约 15.8GB)

# 高质量配置 (768x768, 48帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_S.gguf \
  -p "a futuristic city with flying cars at night" \
  -W 768 -H 768 \
  -f 48 \
  --steps 40 \
  --sampler euler_a \
  --cfg 7.5 \
  --fps 30 \
  -o city.mp4

# 电影配置 (1024x576, 64帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_S.gguf \
  -p "a whale swimming in ocean with bubbles" \
  -W 1024 -H 576 \
  -f 64 \
  --steps 45 \
  --sampler ddim \
  --cfg 8.0 \
  --fps 30 \
  -o whale.mp4

# 超高配置 (1280x720, 32帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_S.gguf \
  -p "explosion in space station" \
  -W 1280 -H 720 \
  -f 32 \
  --steps 50 \
  --sampler ddim \
  --cfg 7.5 \
  -o explosion.mp4
```

### 环境变量

```bash
# 24GB 显存优化
export VIDEO_BACKEND=cuda
export VIDEO_VRAM_SIZE_MB=24576
export VIDEO_AVAILABLE_MEMORY_MB=24576
export VIDEO_STEPS=40
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=euler_a
export VIDEO_USE_FLASH_ATTENTION=true
export VIDEO_NUM_THREADS=8
```

### 性能预期
- CUDA: 约 20-40 秒
- 内存占用: ~22GB

---

## 32GB 显存配置

### 特点
- Q5_K_M 量化流畅
- 2K 分辨率支持
- 长视频生成

### 推荐配置

```bash
# 使用 Q5_K_M 量化版本 (约 16.1GB)

# 超高质量 (1024x1024, 64帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_M.gguf \
  -p "a detailed fantasy castle on mountain with waterfall" \
  -W 1024 -H 1024 \
  -f 64 \
  --steps 50 \
  --sampler ddim \
  --cfg 8.0 \
  --fps 30 \
  -o castle.mp4

# 电影级 (1280x768, 96帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_M.gguf \
  -p "a submarine exploring deep ocean with bioluminescent creatures" \
  -W 1280 -H 768 \
  -f 96 \
  --steps 50 \
  --sampler dpm++ \
  --cfg 7.5 \
  --fps 24 \
  -o submarine.mp4

# 最大配置 (1536x864, 48帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_M.gguf \
  -p "formation of planets in space with stars" \
  -W 1536 -H 864 \
  -f 48 \
  --steps 60 \
  --sampler ddim \
  --cfg 8.0 \
  -o planets.mp4
```

### 环境变量

```bash
# 32GB 显存优化
export VIDEO_BACKEND=cuda
export VIDEO_VRAM_SIZE_MB=32768
export VIDEO_AVAILABLE_MEMORY_MB=32768
export VIDEO_STEPS=50
export VIDEO_GUIDANCE_SCALE=8.0
export VIDEO_SAMPLER=ddim
export VIDEO_USE_FLASH_ATTENTION=true
export VIDEO_NUM_THREADS=12

# 启用模型卸载
export VIDEO_USE_OFFLOAD=true
export VIDEO_OFFLOAD_THRESHOLD_MB=24000
```

### 性能预期
- CUDA: 约 15-30 秒
- 内存占用: ~28GB

---

## 48GB 显存配置

### 特点
- Q8_0 量化可用
- 极高质量输出
- 超长视频

### 推荐配置

```bash
# 使用 Q8_0 量化版本 (约 22.8GB)

# 极致质量 (1280x1280, 96帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q8_0.gguf \
  -p "photorealistic waterfall with rainbow mist in tropical forest" \
  -W 1280 -H 1280 \
  -f 96 \
  --steps 50 \
  --sampler dpm++ \
  --cfg 7.5 \
  --fps 30 \
  -o waterfall_8k.mp4

# 超长视频 (1600x900, 128帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q8_0.gguf \
  -p "time-lapse of city from day to night with traffic lights" \
  -W 1600 -H 900 \
  -f 128 \
  --steps 60 \
  --sampler ddim \
  --cfg 8.0 \
  --fps 24 \
  -o city_timelapse.mp4

# 8K 预览 (2048x1152, 24帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q8_0.gguf \
  -p "alien planet landscape with two suns" \
  -W 2048 -H 1152 \
  -f 24 \
  --steps 80 \
  --sampler dpm++ \
  --cfg 7.5 \
  -o alien.mp4
```

### 环境变量

```bash
# 48GB 显存优化
export VIDEO_BACKEND=cuda
export VIDEO_VRAM_SIZE_MB=49152
export VIDEO_AVAILABLE_MEMORY_MB=49152
export VIDEO_STEPS=50
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=dpm++
export VIDEO_USE_FLASH_ATTENTION=true
export VIDEO_NUM_THREADS=16
export VIDEO_CUDA_GPU_IDS=0

# 性能优化
export CUDA_MODULE_LOADING=LAZY
```

### 性能预期
- CUDA: 约 10-20 秒
- 内存占用: ~45GB

---

## 96GB 显存配置

### 特点
- 多 GPU 支持
- 全精度模型
- 最高质量

### 推荐配置

```bash
# 使用 BF16/FP16 模型 或多卡 Q8_0

# 极限质量 (2048x2048, 128帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-BF16.gguf \
  -p "cinematic shot of dragon breathing fire in slow motion" \
  -W 2048 -H 2048 \
  -f 128 \
  --steps 80 \
  --sampler dpm++ \
  --cfg 7.5 \
  --fps 48 \
  -o dragon_fire.mp4

# 超8K (3072x1728, 64帧)
./bin/video \
  -m models/ltx-2.3-22b-dev-BF16.gguf \
  -p "spaceship battle in asteroid field" \
  -W 3072 -H 1728 \
  -f 64 \
  --steps 100 \
  --sampler ddim \
  --cfg 8.0 \
  --fps 30 \
  -o space_battle.mp4

# 多 GPU 配置 (2x 48GB)
./bin/video \
  -m models/ltx-2.3-22b-dev-Q8_0.gguf \
  -p "AI generated movie scene" \
  -W 2560 -H 1440 \
  -f 96 \
  --steps 60 \
  --sampler dpm++ \
  --cfg 7.5 \
  -o movie.mp4
```

### 环境变量

```bash
# 96GB 显存优化 (多 GPU)
export VIDEO_BACKEND=cuda
export VIDEO_VRAM_SIZE_MB=98304
export VIDEO_AVAILABLE_MEMORY_MB=98304
export VIDEO_STEPS=80
export VIDEO_GUIDANCE_SCALE=7.5
export VIDEO_SAMPLER=dpm++

# 多 GPU 配置
export CUDA_VISIBLE_DEVICES=0,1
export VIDEO_CUDA_GPU_IDS=0,1

# 启用所有优化
export VIDEO_USE_FLASH_ATTENTION=true
export VIDEO_NUM_THREADS=24

# 分布式训练 (未来)
# export VIDEO_DISTRIBUTED=true
```

### 性能预期
- 单卡 CUDA: 约 8-15 秒
- 双卡 CUDA: 约 5-10 秒
- 内存占用: ~90GB

---

## 批量生成配置

### 并行生成 (多文件)

```bash
# 生成多个视频
for prompt in "a cat running" "a dog swimming" "a bird flying"; do
  ./bin/video \
    -m models/ltx-2.3-22b-dev-Q4_K_M.gguf \
    -p "$prompt" \
    -W 512 -H 512 \
    -f 16 \
    --steps 20 \
    -o "${prompt// /_}.mp4" &
done
wait
```

### 高效测试配置 (8GB)

```bash
# 快速测试
./bin/video \
  -m models/ltx-2.3-22b-dev-Q4_K_S.gguf \
  -p "test prompt" \
  -W 256 -H 256 \
  -f 9 \
  --steps 10 \
  --seed 42
```

### 生产级配置 (32GB+)

```bash
# 高吞吐配置
./bin/video \
  -m models/ltx-2.3-22b-dev-Q5_K_M.gguf \
  -p "production video" \
  -W 1024 -H 576 \
  -f 64 \
  --steps 45 \
  --sampler ddim \
  --cfg 7.5 \
  --fps 30 \
  --seed $(date +%s) \
  -o production.mp4
```

---

## 显存优化技巧

### 1. 量化选择

```
8GB  → Q4_K_S (13.7GB)
12GB → Q4_K_M (15.1GB)
16GB → Q4_K_M (15.1GB)
24GB → Q5_K_S (15.8GB)
32GB → Q5_K_M (16.1GB)
48GB → Q8_0 (22.8GB)
96GB → BF16 (42GB)
```

### 2. 分辨率计算

```python
# 显估算公式
可用显存_GB * 0.7 = 模型占用
剩余显存 = 总显存 - 模型占用
最大像素 = 剩余显存 * 1024 * 1024 / 3  # 3通道 RGB

# 示例: 16GB 显存
# 模型: ~14GB
# 剩余: ~2GB
# 最大像素: ~700万像素
# 推荐: 1024x1024 (约100万像素)
```

### 3. 帧数计算

```python
# 帧数估算
latent_frames = frames / 4  # VAE 压缩
latent_channels = 16
latent_h = height / 8
latent_w = width / 8

latent_size = latent_frames * latent_channels * latent_h * latent_w * 4  # float32
total_vram = 模型 + latent_size + 中间缓存
```

### 4. 混合精度

```bash
# 启用 FP16 加速
export VIDEO_USE_FP16=true
export VIDEO_CUDNN_BENCHMARK=true
```

---

## 快速参考表

| 显存 | 模型 | 命令 |
|------|------|------|
| 8GB | Q4_K_S | `-W 384 -H 384 -f 16 --steps 20` |
| 12GB | Q4_K_M | `-W 512 -H 512 -f 24 --steps 25` |
| 16GB | Q4_K_M | `-W 640 -H 640 -f 32 --steps 30` |
| 24GB | Q5_K_S | `-W 768 -H 768 -f 48 --steps 40` |
| 32GB | Q5_K_M | `-W 1024 -H 1024 -f 64 --steps 50` |
| 48GB | Q8_0 | `-W 1280 -H 1280 -f 96 --steps 50` |
| 96GB | BF16 | `-W 2048 -H 2048 -f 128 --steps 80` |

---

## 自动检测脚本

```bash
#!/bin/bash
# detect_vram.sh - 自动检测显存并选择最佳配置

# 检测显存 (Linux)
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
elif [ -f /proc/meminfo ]; then
    VRAM_MB=$(grep MemTotal /proc/meminfo | awk '{print $2/1024}')
else
    echo "Cannot detect memory"
    exit 1
fi

echo "Detected VRAM: ${VRAM_MB}MB"

# 根据显存选择配置
if [ $VRAM_MB -lt 10000 ]; then
    echo "8GB configuration"
    QUANT="Q4_K_S"
    W=384; H=384; F=16; STEPS=20
elif [ $VRAM_MB -lt 17000 ]; then
    echo "12GB configuration"
    QUANT="Q4_K_M"
    W=512; H=512; F=24; STEPS=25
elif [ $VRAM_MB -lt 20000 ]; then
    echo "16GB configuration"
    QUANT="Q4_K_M"
    W=640; H=640; F=32; STEPS=30
elif [ $VRAM_MB -lt 35000 ]; then
    echo "24GB configuration"
    QUANT="Q5_K_S"
    W=768; H=768; F=48; STEPS=40
elif [ $VRAM_MB -lt 55000 ]; then
    echo "32GB configuration"
    QUANT="Q5_K_M"
    W=1024; H=1024; F=64; STEPS=50
else
    echo "48GB+ configuration"
    QUANT="Q8_0"
    W=1280; H=1280; F=96; STEPS=50
fi

# 运行
./bin/video \
  -m "models/ltx-2.3-22b-dev-${QUANT}.gguf" \
  -p "$1" \
  -W $W -H $H -f $F \
  --steps $STEPS \
  -o "output.mp4"
```