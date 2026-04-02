pub mod backend;
pub mod config;
pub mod encoder;
pub mod libcore;
pub mod model;
pub mod pipeline;
pub mod scheduler;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use config::Config;
pub use libcore::*;

use std::path::Path;

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaBackend;

#[cfg(feature = "vulkan")]
use crate::backend::vulkan::VulkanBackend;

use crate::backend::cpu::CpuBackend;

pub fn create_backend_from_env() -> std::sync::Arc<dyn crate::libcore::traits::Backend> {
    let backend_str = std::env::var("VIDEO_BACKEND").unwrap_or_else(|_| "auto".into());
    let device_id = std::env::var("VIDEO_DEVICE_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    match backend_str.to_lowercase().as_str() {
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                match CudaBackend::new(device_id) {
                    Ok(b) => return std::sync::Arc::new(b),
                    Err(e) => {
                        tracing::warn!("Failed to create CUDA backend: {}, falling back to CPU", e);
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA support not compiled in, using CPU backend");
            }
        }
        "vulkan" => {
            #[cfg(feature = "vulkan")]
            {
                match VulkanBackend::new(device_id) {
                    Ok(b) => return std::sync::Arc::new(b),
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create Vulkan backend: {}, falling back to CPU",
                            e
                        );
                    }
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                tracing::warn!("Vulkan support not compiled in, using CPU backend");
            }
        }
        _ => {}
    }

    std::sync::Arc::new(CpuBackend::new())
}

pub fn init_logging() {
    let log_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into());

    let _ = tracing_subscriber::fmt()
        .with_max_level(match log_level.to_lowercase().as_str() {
            "trace" => tracing::Level::TRACE,
            "debug" => tracing::Level::DEBUG,
            "info" => tracing::Level::INFO,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        })
        .with_target(false)
        .try_init();
}

pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    init_logging();

    let config = Config::from_env();

    tracing::info!("video.cpp v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Backend: {:?}", config.backend.backend_type);
    tracing::info!("Model path: {}", config.model.model_path);

    if !Path::new(&config.model.model_path).exists() {
        tracing::warn!("Model file not found: {}", config.model.model_path);
        tracing::info!("Please set VIDEO_MODEL_PATH to point to your .gguv model file");
        return Ok(());
    }

    Ok(())
}
