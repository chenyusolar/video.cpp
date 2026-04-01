use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Config {
    pub backend: BackendConfig,
    pub model: ModelConfig,
    pub generation: GenerationConfig,
    pub memory: MemoryConfig,
    pub quantization: QuantizationConfig,
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub backend_type: BackendType,
    pub device_id: usize,
    pub use_flash_attention: bool,
    pub num_threads: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum BackendType {
    Auto,
    CPU,
    CUDA,
    Vulkan,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Auto,
            device_id: 0,
            use_flash_attention: true,
            num_threads: num_cpus::get(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: String,
    pub model_type: String,
    pub latent_channels: u32,
    pub in_channels: u32,
    pub out_channels: u32,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub steps: u32,
    pub guidance_scale: f32,
    pub sampler: SamplerType,
    pub fps: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum SamplerType {
    Euler,
    EulerA,
    DDIM,
    DPMPlusPlus,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            steps: 30,
            guidance_scale: 7.5,
            sampler: SamplerType::Euler,
            fps: 24,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub vram_size_mb: u64,
    pub available_memory_mb: u64,
    pub offload_threshold_mb: u64,
    pub use_offload: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            vram_size_mb: 0,
            available_memory_mb: 0,
            offload_threshold_mb: 16000,
            use_offload: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub quant_type: QuantType,
    pub block_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum QuantType {
    None,
    Int8,
    NF4,
    FP8,
    Auto,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantType::Auto,
            block_size: 32,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            backend: BackendConfig {
                backend_type: match std::env::var("VIDEO_BACKEND")
                    .unwrap_or_else(|_| "auto".into())
                    .to_lowercase()
                    .as_str()
                {
                    "cpu" => BackendType::CPU,
                    "cuda" => BackendType::CUDA,
                    "vulkan" => BackendType::Vulkan,
                    _ => BackendType::Auto,
                },
                device_id: std::env::var("VIDEO_DEVICE_ID")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                use_flash_attention: std::env::var("VIDEO_USE_FLASH_ATTENTION")
                    .map(|s| s.to_lowercase() != "false" && s != "0")
                    .unwrap_or(true),
                num_threads: std::env::var("VIDEO_NUM_THREADS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(num_cpus::get()),
            },
            model: ModelConfig {
                model_path: std::env::var("VIDEO_MODEL_PATH")
                    .unwrap_or_else(|_| "models/ltx2.gguv".into()),
                model_type: std::env::var("VIDEO_MODEL_TYPE")
                    .unwrap_or_else(|_| "ltx-video".into()),
                latent_channels: 16,
                in_channels: 16,
                out_channels: 16,
            },
            generation: GenerationConfig {
                steps: std::env::var("VIDEO_STEPS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30),
                guidance_scale: std::env::var("VIDEO_GUIDANCE_SCALE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(7.5),
                sampler: match std::env::var("VIDEO_SAMPLER")
                    .unwrap_or_else(|_| "euler".into())
                    .to_lowercase()
                    .as_str()
                {
                    "eulera" | "euler_a" => SamplerType::EulerA,
                    "ddim" => SamplerType::DDIM,
                    "dpm++" | "dpmpp" => SamplerType::DPMPlusPlus,
                    _ => SamplerType::Euler,
                },
                fps: std::env::var("VIDEO_FPS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(24),
            },
            memory: MemoryConfig {
                vram_size_mb: std::env::var("VIDEO_VRAM_SIZE_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                available_memory_mb: std::env::var("VIDEO_AVAILABLE_MEMORY_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                offload_threshold_mb: std::env::var("VIDEO_OFFLOAD_THRESHOLD_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(16000),
                use_offload: std::env::var("VIDEO_USE_OFFLOAD")
                    .map(|s| s.to_lowercase() == "true" || s == "1")
                    .unwrap_or(false),
            },
            quantization: QuantizationConfig {
                quant_type: match std::env::var("VIDEO_QUANTIZATION")
                    .unwrap_or_else(|_| "auto".into())
                    .to_lowercase()
                    .as_str()
                {
                    "none" | "float" | "fp32" => QuantType::None,
                    "int8" | "int" => QuantType::Int8,
                    "nf4" | "n4" => QuantType::NF4,
                    "fp8" | "fp" => QuantType::FP8,
                    _ => QuantType::Auto,
                },
                block_size: std::env::var("VIDEO_QUANT_BLOCK_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(32),
            },
        }
    }

    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "VIDEO_BACKEND".into(),
            format!("{:?}", self.backend.backend_type),
        );
        map.insert("VIDEO_DEVICE_ID".into(), self.backend.device_id.to_string());
        map.insert(
            "VIDEO_USE_FLASH_ATTENTION".into(),
            self.backend.use_flash_attention.to_string(),
        );
        map.insert(
            "VIDEO_NUM_THREADS".into(),
            self.backend.num_threads.to_string(),
        );
        map.insert("VIDEO_MODEL_PATH".into(), self.model.model_path.clone());
        map.insert("VIDEO_MODEL_TYPE".into(), self.model.model_type.clone());
        map.insert("VIDEO_STEPS".into(), self.generation.steps.to_string());
        map.insert(
            "VIDEO_GUIDANCE_SCALE".into(),
            self.generation.guidance_scale.to_string(),
        );
        map.insert("VIDEO_FPS".into(), self.generation.fps.to_string());
        map.insert(
            "VIDEO_VRAM_SIZE_MB".into(),
            self.memory.vram_size_mb.to_string(),
        );
        map.insert(
            "VIDEO_AVAILABLE_MEMORY_MB".into(),
            self.memory.available_memory_mb.to_string(),
        );
        map.insert(
            "VIDEO_OFFLOAD_THRESHOLD_MB".into(),
            self.memory.offload_threshold_mb.to_string(),
        );
        map.insert(
            "VIDEO_USE_OFFLOAD".into(),
            self.memory.use_offload.to_string(),
        );
        map
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_env()
    }
}
