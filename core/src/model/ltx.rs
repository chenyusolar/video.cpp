use std::path::Path;

use crate::config::Config;
use crate::libcore::backend::Backend;
use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Device, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error as CoreError, Model, Result as CoreResult, TextEncoder, VAE};
use crate::model::loader::GGUFVIDLoader;
use crate::scheduler::{DiffusionScheduler, SchedulerType};
use std::sync::Arc;

pub struct LTXModel {
    loader: Arc<GGUFVIDLoader>,
    transformer_weights: std::collections::HashMap<String, Tensor>,
    vae_weights: std::collections::HashMap<String, Tensor>,
    text_encoder_weights: std::collections::HashMap<String, Tensor>,
    latent_channels: u32,
    in_channels: u32,
    out_channels: u32,
}

impl LTXModel {
    pub fn new(loader: &GGUFVIDLoader) -> CoreResult<Self> {
        let metadata = loader.metadata();

        let mut model = Self {
            loader: Arc::new(loader.clone()),
            transformer_weights: std::collections::HashMap::new(),
            vae_weights: std::collections::HashMap::new(),
            text_encoder_weights: std::collections::HashMap::new(),
            latent_channels: metadata.latent_channels,
            in_channels: metadata.latent_channels * 2,
            out_channels: metadata.latent_channels,
        };

        model.load_weights()?;

        Ok(model)
    }

    fn load_weights(&mut self) -> CoreResult<()> {
        let tensors = self.loader.list_tensors();

        for tensor_info in tensors {
            let tensor = self.loader.load_tensor(&tensor_info.name)?;

            if tensor_info.name.starts_with("transformer.") {
                self.transformer_weights
                    .insert(tensor_info.name.clone(), tensor);
            } else if tensor_info.name.starts_with("vae.") {
                self.vae_weights.insert(tensor_info.name.clone(), tensor);
            } else if tensor_info.name.starts_with("text_encoder.") {
                self.text_encoder_weights
                    .insert(tensor_info.name.clone(), tensor);
            }
        }

        Ok(())
    }

    pub fn get_transformer_weight(&self, name: &str) -> CoreResult<&Tensor> {
        self.transformer_weights
            .get(name)
            .ok_or_else(|| CoreError::NotFound(format!("Weight not found: {}", name)))
    }

    pub fn get_vae_weight(&self, name: &str) -> CoreResult<&Tensor> {
        self.vae_weights
            .get(name)
            .ok_or_else(|| CoreError::NotFound(format!("VAE weight not found: {}", name)))
    }
}

impl Model for LTXModel {
    fn forward(&self, latent: &Tensor, timestep: f32, context: &Context) -> CoreResult<Tensor> {
        let timestep_emb = self.get_timestep_embedding(timestep)?;

        let hidden_states = latent.clone();

        let temb = self.time_embedding(&timestep_emb)?;

        let mut x = hidden_states;

        x = self.apply_transformer_block(&x, &temb, context)?;

        let noise_pred = x;

        Ok(noise_pred)
    }

    fn parameters(&self) -> usize {
        self.transformer_weights
            .values()
            .map(|t| t.shape().volume() as usize)
            .sum()
    }
}

impl LTXModel {
    fn get_timestep_embedding(&self, timestep: f32) -> CoreResult<Tensor> {
        let dim = 256;
        let half_dim = dim / 2;

        let emb = (0..half_dim)
            .map(|i| {
                let freq = (-0.34657359_f32 * (i as f32) / (half_dim as f32)).exp();
                let arg = timestep * freq * std::f32::consts::PI;
                (arg.sin(), arg.cos())
            })
            .collect::<Vec<_>>();

        let mut embedding = Vec::with_capacity(dim);
        for (sin, cos) in emb {
            embedding.push(sin);
            embedding.push(cos);
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, dim as u32]),
            TensorData::F32(embedding),
        ))
    }

    fn time_embedding(&self, timestep_emb: &Tensor) -> CoreResult<Tensor> {
        let out = timestep_emb.clone();
        Ok(out)
    }

    fn apply_transformer_block(
        &self,
        x: &Tensor,
        temb: &Tensor,
        context: &Context,
    ) -> CoreResult<Tensor> {
        let x = x.clone();
        Ok(x)
    }
}

impl VAE for LTXModel {
    fn encode(&self, pixels: &Tensor) -> CoreResult<Tensor> {
        let shape = pixels.shape();
        if shape.len() != 5 {
            return Err(CoreError::Tensor(
                "encode requires 5D tensor [B,T,C,H,W]".into(),
            ));
        }

        let [B, T, C, H, W] = [shape[0], shape[1], shape[2], shape[3], shape[4]];

        let latent_h = H / 8;
        let latent_w = W / 8;

        let mut latent_data =
            vec![0.0_f32; (B * T * self.latent_channels * latent_h * latent_w) as usize];

        for i in 0..latent_data.len() {
            latent_data[i] = (i as f32 * 0.01).sin();
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, T, self.latent_channels, latent_h, latent_w]),
            TensorData::F32(latent_data),
        ))
    }

    fn decode(&self, latent: &Tensor) -> CoreResult<Tensor> {
        let shape = latent.shape();
        if shape.len() != 5 {
            return Err(CoreError::Tensor(
                "decode requires 5D tensor [B,T,C,H,W]".into(),
            ));
        }

        let [B, T, C, H, W] = [shape[0], shape[1], shape[2], shape[3], shape[4]];

        let frames_h = H * 8;
        let frames_w = W * 8;
        let channels = 3;

        let mut frame_data = vec![0.0_f32; (B * T * channels * frames_h * frames_w) as usize];

        for i in 0..frame_data.len() {
            let t = (i as f32 * 0.001).sin() * 0.5 + 0.5;
            frame_data[i] = t;
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, T, channels, frames_h, frames_w]),
            TensorData::F32(frame_data),
        ))
    }
}

impl TextEncoder for LTXModel {
    fn encode(&self, text: &str) -> CoreResult<Context> {
        let text_embeddings = self.encode_text_simple(text)?;

        Ok(Context {
            embeddings: text_embeddings,
            embeddings_neg: None,
            seq_len: text.len() as u32 / 4,
        })
    }

    fn encode_negative(&self, text: &str) -> CoreResult<Context> {
        let text_embeddings = self.encode_text_simple(text)?;

        Ok(Context {
            embeddings: text_embeddings,
            embeddings_neg: None,
            seq_len: text.len() as u32 / 4,
        })
    }
}

impl LTXModel {
    fn encode_text_simple(&self, text: &str) -> CoreResult<Tensor> {
        let seq_len = 77_u32;
        let hidden_size = 768_u32;

        let num_tokens = text.len().min(seq_len as usize);

        let mut embeddings = Vec::with_capacity((seq_len * hidden_size) as usize);
        for i in 0..(seq_len * hidden_size) as usize {
            let freq = i as f32 * 0.01;
            embeddings.push(freq.sin().cos());
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, seq_len, hidden_size]),
            TensorData::F32(embeddings),
        ))
    }
}
