pub mod traits;
pub mod cpu;
pub mod cuda;
pub mod vulkan;

pub use traits::Backend;
pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "vulkan")]
pub use vulkan::VulkanBackend;

pub fn create_backend(name: &str, device_id: usize) -> Result<std::sync::Arc<dyn Backend>, crate::Error> {
    match name.to_lowercase().as_str() {
        "cpu" => Ok(std::sync::Arc::new(CpuBackend::new()) as std::sync::Arc<dyn Backend>),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(std::sync::Arc::new(CudaBackend::new(device_id)?) as std::sync::Arc<dyn Backend>),
        #[cfg(feature = "vulkan")]
        "vulkan" => Ok(std::sync::Arc::new(VulkanBackend::new(device_id)?) as std::sync::Arc<dyn Backend>),
        _ => Err(crate::Error::Backend(format!("Unknown backend: {}", name))),
    }
}
