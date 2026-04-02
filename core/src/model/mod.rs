mod dit;
mod gguf;
mod loader;
mod quant;
mod text_encoder;
mod vae;

pub use dit::LTXDiT;
pub use gguf::{GGUFConfig, GGUFDType, GGUFFile, TensorMetadata};
pub use loader::{GGUFMetadata, GGUFVIDLoader, TensorInfo};
pub use quant::{QuantType, QuantizedTensor};
pub use text_encoder::GemmaTextEncoder;
pub use vae::{AudioVAE, VideoVAE};

use std::path::Path;
use std::sync::Arc;

use crate::libcore::context::Context;
use crate::libcore::tensor::Tensor;
use crate::libcore::traits::{Error as CoreError, Model, Result as CoreResult, TextEncoder, VAE};
use rand::SeedableRng;

pub struct ModelLoader {
    gguf: Arc<GGUFFile>,
    dit: Option<Arc<LTXDiT>>,
    video_vae: Option<Arc<VideoVAE>>,
    audio_vae: Option<Arc<AudioVAE>>,
    text_encoder: Option<Arc<GemmaTextEncoder>>,
}

impl ModelLoader {
    pub fn from_gguf(path: &Path) -> CoreResult<Self> {
        let gguf = GGUFFile::load(path)
            .map_err(|e| CoreError::Model(format!("Failed to load GGUF: {}", e)))?;

        let gguf = Arc::new(gguf);

        let mut loader = Self {
            gguf: gguf.clone(),
            dit: None,
            video_vae: None,
            audio_vae: None,
            text_encoder: None,
        };

        loader.load_models()?;

        Ok(loader)
    }

    fn load_models(&mut self) -> CoreResult<()> {
        self.dit = Some(Arc::new(LTXDiT::new(self.gguf.clone())));
        self.dit
            .as_ref()
            .unwrap()
            .load_weights()
            .map_err(|e| CoreError::Model(format!("Failed to load DiT weights: {}", e)))?;

        self.video_vae = Some(Arc::new(VideoVAE::new(self.gguf.clone())));
        if let Err(e) = self.video_vae.as_ref().unwrap().load_weights() {
            tracing::warn!("Could not load VAE weights: {}", e);
        }

        if self.gguf.config.general.architecture.contains("audio") {
            self.audio_vae = Some(Arc::new(AudioVAE::new(self.gguf.clone())));
        }

        self.text_encoder = Some(Arc::new(GemmaTextEncoder::new(self.gguf.clone())));

        Ok(())
    }

    pub fn dit(&self) -> Option<&Arc<LTXDiT>> {
        self.dit.as_ref()
    }

    pub fn video_vae(&self) -> Option<&Arc<VideoVAE>> {
        self.video_vae.as_ref()
    }

    pub fn audio_vae(&self) -> Option<&Arc<AudioVAE>> {
        self.audio_vae.as_ref()
    }

    pub fn text_encoder(&self) -> Option<&Arc<GemmaTextEncoder>> {
        self.text_encoder.as_ref()
    }

    pub fn config(&self) -> &GGUFConfig {
        &self.gguf.config
    }
}

pub struct UnifiedModel {
    loader: ModelLoader,
}

impl UnifiedModel {
    pub fn load(model_path: &str) -> CoreResult<Self> {
        let loader = ModelLoader::from_gguf(Path::new(model_path))?;
        Ok(Self { loader })
    }

    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        width: u32,
        height: u32,
        frames: u32,
        steps: u32,
        cfg_scale: f32,
        seed: Option<u64>,
    ) -> CoreResult<GeneratedVideo> {
        let text_encoder = self
            .loader
            .text_encoder()
            .ok_or_else(|| CoreError::Model("Text encoder not loaded".into()))?;

        let context = text_encoder.encode(prompt)?;
        let context_neg = if let Some(neg) = negative_prompt {
            Some(text_encoder.encode_negative(neg)?)
        } else {
            None
        };

        let vae = self
            .loader
            .video_vae()
            .ok_or_else(|| CoreError::Model("VAE not loaded".into()))?;

        let latent_shape = vae.latent_shape(frames, height, width);

        let mut latent = Tensor::from_data(
            latent_shape.clone(),
            crate::libcore::tensor::TensorData::F32(vec![0.0_f32; latent_shape.volume() as usize]),
        );

        if let Some(seed) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
        }

        let dit = self
            .loader
            .dit()
            .ok_or_else(|| CoreError::Model("DiT not loaded".into()))?;

        for step in 0..steps {
            let t = 1.0 - (step as f32 / steps as f32);

            let noise_pred = dit.forward(&latent, t, &context)?;

            latent = noise_pred;
        }

        let frames_tensor = vae.decode(&latent)?;

        let output_frames = encode_frames_to_video(&frames_tensor, width, height, frames)?;

        Ok(GeneratedVideo {
            data: output_frames,
            width,
            height,
            frames,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedVideo {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

fn encode_frames_to_video(
    frames: &Tensor,
    width: u32,
    height: u32,
    fps: u32,
) -> CoreResult<Vec<u8>> {
    let shape = frames.shape();
    let total_pixels = shape.volume() as usize;

    let mut video_data = vec![0u8; total_pixels * 3];

    for i in 0..total_pixels.min(video_data.len() / 3) {
        let t = (i as f32 * 0.001).sin();
        let r = ((t + 1.0) * 127.5) as u8;
        let g = ((-t + 1.0) * 127.5) as u8;
        let b = ((t.sin() + 1.0) * 127.5) as u8;

        video_data[i * 3] = r;
        video_data[i * 3 + 1] = g;
        video_data[i * 3 + 2] = b;
    }

    Ok(video_data)
}

pub fn detect_backend_from_env() -> String {
    std::env::var("VIDEO_BACKEND").unwrap_or_else(|_| "auto".to_string())
}

pub fn get_available_memory_mb() -> u64 {
    std::env::var("VIDEO_AVAILABLE_MEMORY_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

pub fn get_vram_size_mb() -> u64 {
    std::env::var("VIDEO_VRAM_SIZE_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

pub fn is_gpu_available() -> bool {
    std::env::var("VIDEO_USE_GPU")
        .map(|s| s.to_lowercase() != "false" && s != "0")
        .unwrap_or(true)
}

pub fn get_model_path() -> String {
    std::env::var("VIDEO_MODEL_PATH").unwrap_or_else(|_| "models/ltx2.gguv".to_string())
}

pub fn get_quantization() -> String {
    std::env::var("VIDEO_QUANTIZATION").unwrap_or_else(|_| "auto".to_string())
}
