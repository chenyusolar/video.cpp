use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Device, Tensor, TensorData, TensorShape};
use crate::libcore::traits::Backend;
use crate::libcore::traits::{
    Error as CoreError, Model, Result as CoreResult, Scheduler, TextEncoder, VAE,
};
use crate::model::{GeneratedVideo, ModelLoader, UnifiedModel};
use crate::scheduler::{DiffusionScheduler, SchedulerType};

pub struct VideoPipeline {
    config: Config,
    loader: ModelLoader,
    scheduler: DiffusionScheduler,
    backend: Arc<dyn Backend>,
    start_time_ms: u64,
}

#[derive(Clone)]
pub struct VideoPipelineOptions {
    pub model_path: String,
    pub backend: Arc<dyn Backend>,
    pub steps: u32,
    pub guidance_scale: f32,
    pub sampler: String,
}

impl VideoPipeline {
    pub fn new(model_path: &str, backend: Arc<dyn Backend>) -> CoreResult<Self> {
        let config = Config::from_env();

        let loader = ModelLoader::from_gguf(Path::new(model_path))
            .map_err(|e| CoreError::Model(format!("Failed to load model: {}", e)))?;

        let scheduler = DiffusionScheduler::from_type_str(
            &config.generation.sampler.to_string().to_lowercase(),
            config.generation.steps,
        );

        Ok(Self {
            config,
            loader,
            scheduler,
            backend,
            start_time_ms: 0,
        })
    }

    pub fn with_options(
        model_path: &str,
        backend: Arc<dyn Backend>,
        options: VideoPipelineOptions,
    ) -> CoreResult<Self> {
        let loader = ModelLoader::from_gguf(Path::new(model_path))
            .map_err(|e| CoreError::Model(format!("Failed to load model: {}", e)))?;

        let scheduler =
            DiffusionScheduler::from_type_str(&options.sampler.to_lowercase(), options.steps);

        let mut config = Config::from_env();
        config.generation.steps = options.steps;
        config.generation.guidance_scale = options.guidance_scale;

        Ok(Self {
            config,
            loader,
            scheduler,
            backend,
            start_time_ms: 0,
        })
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> CoreResult<VideoOutput> {
        self.start_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let timesteps: Vec<f32> = self.scheduler.timesteps();
        let num_inference_steps = request.steps.unwrap_or(self.config.generation.steps) as usize;
        self.scheduler.set_timesteps(num_inference_steps as u32);
        let timesteps = self.scheduler.timesteps();

        let text_encoder = self
            .loader
            .text_encoder()
            .ok_or_else(|| CoreError::Model("Text encoder not available".into()))?;

        let context = text_encoder.encode(&request.prompt)?;
        let context_neg = if let Some(neg) = &request.negative_prompt {
            Some(text_encoder.encode_negative(neg)?)
        } else {
            None
        };

        let vae = self
            .loader
            .video_vae()
            .ok_or_else(|| CoreError::Model("VAE not available".into()))?;

        let latent_shape = vae.latent_shape(
            request.frames as u32,
            request.height as u32,
            request.width as u32,
        );

        let mut latent = self.backend.randn(latent_shape)?;

        let dit = self
            .loader
            .dit()
            .ok_or_else(|| CoreError::Model("DiT model not available".into()))?;

        let total_steps = timesteps.len();
        for (i, t) in timesteps.iter().enumerate() {
            let t_scaled = *t;

            let noise_pred_uncond = dit.forward(&latent, t_scaled, &context)?;

            let noise_pred_cond = if context_neg.is_some() {
                if let Some(ref ctx_neg) = context_neg {
                    dit.forward(&latent, t_scaled, ctx_neg)?
                } else {
                    latent.clone()
                }
            } else {
                latent.clone()
            };

            let cfg_scale = request
                .cfg_scale
                .unwrap_or(self.config.generation.guidance_scale);
            let noise_pred = self.apply_cfg(&noise_pred_cond, &noise_pred_uncond, cfg_scale)?;

            latent = self.scheduler.step(&latent, t_scaled, &noise_pred)?;

            if let Some(callback) = &request.callback {
                callback(i + 1, total_steps);
            }
        }

        let frames = vae.decode(&latent)?;

        let video_bytes = self.encode_video(&frames, request.fps.unwrap_or(24) as u32)?;

        let end_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(VideoOutput {
            frames: video_bytes,
            width: request.width,
            height: request.height,
            fps: request.fps.unwrap_or(24),
            generation_time_ms: end_time_ms - self.start_time_ms,
        })
    }

    fn apply_cfg(
        &self,
        pred_cond: &Tensor,
        pred_uncond: &Tensor,
        cfg_scale: f32,
    ) -> CoreResult<Tensor> {
        let diff = pred_cond.sub(pred_uncond)?;
        let cfg_tensor =
            Tensor::from_data(pred_cond.shape().clone(), TensorData::F32Scalar(cfg_scale));
        let scaled_diff = diff.mul(&cfg_tensor)?;
        pred_uncond.add(&scaled_diff)
    }

    fn encode_video(&self, frames: &Tensor, fps: u32) -> CoreResult<Vec<u8>> {
        let shape = frames.shape();

        let data = match frames.data() {
            crate::libcore::tensor::TensorData::F32(arr) => arr.clone(),
            _ => {
                return Err(CoreError::Tensor(
                    "Unsupported tensor data type for video encoding".into(),
                ));
            }
        };

        let mut video_data = Vec::with_capacity(data.len());

        for &value in &data {
            let scaled = ((value.clamp(0.0, 1.0) * 255.0) as u8);
            video_data.push(scaled);
        }

        Ok(video_data)
    }

    pub fn image_to_video(
        &mut self,
        init_image: &[u8],
        prompt: &str,
        width: u32,
        height: u32,
        frames: u32,
        strength: f32,
        steps: u32,
        cfg_scale: f32,
        seed: Option<u64>,
    ) -> CoreResult<VideoOutput> {
        self.start_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.scheduler.set_timesteps(steps);
        let timesteps = self.scheduler.timesteps();

        let text_encoder = self
            .loader
            .text_encoder()
            .ok_or_else(|| CoreError::Model("Text encoder not available".into()))?;
        let context = text_encoder.encode(prompt)?;

        let vae = self
            .loader
            .video_vae()
            .ok_or_else(|| CoreError::Model("VAE not available".into()))?;

        let latent_shape = vae.latent_shape(frames, height, width);

        let mut latent = self.backend.randn(latent_shape)?;

        let dit = self
            .loader
            .dit()
            .ok_or_else(|| CoreError::Model("DiT model not available".into()))?;

        let _total_steps = timesteps.len();
        for (i, t) in timesteps.iter().enumerate() {
            let noise_pred = dit.forward(&latent, *t, &context)?;
            latent = self.scheduler.step(&latent, *t, &noise_pred)?;

            let _ = i;
        }

        let frames_tensor = vae.decode(&latent)?;
        let video_bytes = self.encode_video(&frames_tensor, 24)?;

        let end_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(VideoOutput {
            frames: video_bytes,
            width: width as usize,
            height: height as usize,
            fps: 24,
            generation_time_ms: end_time_ms - self.start_time_ms,
        })
    }

    pub fn video_to_video(
        &mut self,
        init_video: &[u8],
        prompt: &str,
        width: u32,
        height: u32,
        frames: u32,
        strength: f32,
        steps: u32,
        cfg_scale: f32,
        seed: Option<u64>,
    ) -> CoreResult<VideoOutput> {
        self.start_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.scheduler.set_timesteps(steps);
        let timesteps = self.scheduler.timesteps();

        let text_encoder = self
            .loader
            .text_encoder()
            .ok_or_else(|| CoreError::Model("Text encoder not available".into()))?;
        let context = text_encoder.encode(prompt)?;

        let vae = self
            .loader
            .video_vae()
            .ok_or_else(|| CoreError::Model("VAE not available".into()))?;

        let latent_shape = vae.latent_shape(frames, height, width);

        let mut latent = self.backend.randn(latent_shape)?;

        let dit = self
            .loader
            .dit()
            .ok_or_else(|| CoreError::Model("DiT model not available".into()))?;

        let _total_steps = timesteps.len();
        for (i, t) in timesteps.iter().enumerate() {
            let noise_pred = dit.forward(&latent, *t, &context)?;
            latent = self.scheduler.step(&latent, *t, &noise_pred)?;

            let _ = i;
        }

        let frames_tensor = vae.decode(&latent)?;
        let video_bytes = self.encode_video(&frames_tensor, 24)?;

        let end_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Ok(VideoOutput {
            frames: video_bytes,
            width: width as usize,
            height: height as usize,
            fps: 24,
            generation_time_ms: end_time_ms - self.start_time_ms,
        })
    }
}

pub struct GenerateRequest<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub fps: Option<usize>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub seed: Option<u64>,
    pub callback: Option<Box<dyn Fn(usize, usize) + Send + 'a>>,
}

impl<'a> Clone for GenerateRequest<'a> {
    fn clone(&self) -> Self {
        GenerateRequest {
            prompt: self.prompt,
            negative_prompt: self.negative_prompt,
            frames: self.frames,
            width: self.width,
            height: self.height,
            fps: self.fps,
            steps: self.steps,
            cfg_scale: self.cfg_scale,
            seed: self.seed,
            callback: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub frames: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub generation_time_ms: u64,
}
