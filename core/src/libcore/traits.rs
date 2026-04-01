use crate::libcore::context::{Context, SamplerType};
use crate::libcore::tensor::{DType, Device, Tensor, TensorShape};
use thiserror::Error;

pub use crate::libcore::context::{GenerateRequest, GenerationStats, VideoOutput};

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tensor error: {0}")]
    Tensor(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Model: Send + Sync {
    fn forward(&self, latent: &Tensor, timestep: f32, context: &Context) -> Result<Tensor>;
    fn parameters(&self) -> usize;
}

pub trait Scheduler: Send + Sync {
    fn step(&mut self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor>;
    fn add_noise(&mut self, latent: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor>;
    fn timesteps(&self) -> Vec<f32>;
    fn set_timesteps(&mut self, num_steps: u32);
}

pub trait VAE: Send + Sync {
    fn encode(&self, pixels: &Tensor) -> Result<Tensor>;
    fn decode(&self, latent: &Tensor) -> Result<Tensor>;
    fn latent_shape(&self, frames: u32, height: u32, width: u32) -> TensorShape {
        let latent_channels = 16u32;
        TensorShape::new(vec![
            1, // batch
            frames,
            height / 8,
            width / 8,
            latent_channels,
        ])
    }
}

pub trait TextEncoder: Send + Sync {
    fn encode(&self, text: &str) -> Result<Context>;
    fn encode_negative(&self, text: &str) -> Result<Context>;
}

pub trait VideoEncoder: Send + Sync {
    fn encode(&self, frames: &Tensor, fps: u32) -> Result<Vec<u8>>;
}

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    fn tensor_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn tensor_conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        opts: super::libcore::tensor::Conv2dOpts,
    ) -> Result<Tensor>;

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor>;
    fn temporal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, t: u32) -> Result<Tensor>;

    fn alloc_tensor(&self, shape: TensorShape, dtype: DType) -> Result<Tensor>;
    fn copy_to_device(&self, data: &[f32], tensor: &mut Tensor) -> Result<()>;
    fn copy_to_host(&self, tensor: &Tensor) -> Result<Vec<f32>>;

    fn randn(&self, shape: TensorShape) -> Result<Tensor>;
    fn randn_like(&self, tensor: &Tensor) -> Result<Tensor>;
    fn zeros(&self, shape: TensorShape, dtype: DType) -> Result<Tensor>;
    fn ones(&self, shape: TensorShape, dtype: DType) -> Result<Tensor>;

    fn synchronize(&self) -> Result<()>;
    fn memory_allocated(&self) -> u64;
    fn memory_reserved(&self) -> u64;
}

pub enum BackendType {
    CPU,
    CUDA(usize),
    Vulkan,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::CPU
    }
}

impl BackendType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" => BackendType::CUDA(0),
            "cuda:0" => BackendType::CUDA(0),
            "cuda:1" => BackendType::CUDA(1),
            "vulkan" | "vulkancompute" => BackendType::Vulkan,
            "cpu" | _ => BackendType::CPU,
        }
    }
}

pub fn create_backend(backend: BackendType) -> std::sync::Arc<dyn Backend> {
    match backend {
        BackendType::CUDA(_) => {
            // Try CUDA backend if available
            #[cfg(feature = "cuda")]
            {
                return std::sync::Arc::new(
                    super::backend::cuda::CudaBackend::new().unwrap_or_else(|e| {
                        tracing::warn!("CUDA not available: {}, falling back to CPU", e);
                        super::backend::cpu::CpuBackend::new() as Arc<dyn Backend>
                    }),
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA support not compiled in, using CPU backend");
                std::sync::Arc::new(super::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>
            }
        }
        BackendType::Vulkan => {
            // Try Vulkan backend if available
            #[cfg(feature = "vulkan")]
            {
                return std::sync::Arc::new(
                    super::backend::vulkan::VulkanBackend::new().unwrap_or_else(|e| {
                        tracing::warn!("Vulkan not available: {}, falling back to CPU", e);
                        super::backend::cpu::CpuBackend::new() as Arc<dyn Backend>
                    }),
                );
            }
            #[cfg(not(feature = "vulkan"))]
            {
                tracing::warn!("Vulkan support not compiled in, using CPU backend");
                std::sync::Arc::new(super::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>
            }
        }
        BackendType::CPU => {
            std::sync::Arc::new(super::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>
        }
    }
}
