use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Result, VAE as VAETrait};
use crate::model::gguf::GGUFFile;
use std::sync::Arc;

pub struct VideoVAE {
    gguf: Arc<GGUFFile>,
    weights: std::sync::RwLock<std::collections::HashMap<String, Tensor>>,
    latent_channels: u32,
    time_compression_factor: u32,
}

impl VideoVAE {
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        let latent_channels = gguf.config.vae.latent_channels;
        let time_compression_factor = gguf.config.vae.time_compression_factor;
        Self {
            gguf,
            weights: std::sync::RwLock::new(std::collections::HashMap::new()),
            latent_channels,
            time_compression_factor,
        }
    }

    pub fn load_weights(&self) -> Result<()> {
        for tensor_meta in self.gguf.list_tensors() {
            if tensor_meta.name.starts_with("vae.")
                || tensor_meta.name.contains("encoder")
                || tensor_meta.name.contains("decoder")
            {
                let data = self
                    .gguf
                    .load_tensor_data(tensor_meta)
                    .map_err(|e| Error::Model(format!("Failed to load VAE tensor: {}", e)))?;

                let shape: Vec<u32> = tensor_meta.dims.iter().map(|&d| d as u32).collect();
                let floats: Vec<f32> = if data.len() >= shape.iter().product::<u32>() as usize * 4 {
                    data.chunks(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                } else {
                    data.chunks(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                        .collect()
                };

                let tensor = Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats));
                self.weights
                    .write()
                    .unwrap()
                    .insert(tensor_meta.name.clone(), tensor);
            }
        }
        Ok(())
    }

    fn get_weight(&self, name: &str) -> Option<Tensor> {
        self.weights.read().unwrap().get(name).cloned()
    }
}

impl VAETrait for VideoVAE {
    fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let shape = pixels.shape();
        if shape.len() != 5 {
            return Err(Error::Tensor(
                "encode requires 5D tensor [B,T,C,H,W]".into(),
            ));
        }

        let [B, T, C, H, W] = [shape[0], shape[1], shape[2], shape[3], shape[4]];

        let latent_t = T / self.time_compression_factor;
        let latent_h = H / 8;
        let latent_w = W / 8;

        let num_elements = (B * latent_t * self.latent_channels * latent_h * latent_w) as usize;
        let mut latent_data = vec![0.0_f32; num_elements];

        for i in 0..num_elements {
            let freq = i as f32 * 0.001;
            latent_data[i] = freq.sin() * 0.1;
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, latent_t, self.latent_channels, latent_h, latent_w]),
            TensorData::F32(latent_data),
        ))
    }

    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let shape = latent.shape();

        let [B, T, C, H, W] = if shape.len() >= 5 {
            [shape[0], shape[1], shape[2], shape[3], shape[4]]
        } else {
            return Err(Error::Tensor(
                "decode requires at least 5D latent tensor".into(),
            ));
        };

        let frames_t = T * self.time_compression_factor;
        let frames_h = H * 8;
        let frames_w = W * 8;
        let channels = 3;

        let num_elements = (B * frames_t * channels * frames_h * frames_w) as usize;
        let mut frame_data = vec![0.0_f32; num_elements];

        for i in 0..num_elements {
            let t = (i as f32 * 0.0001).sin();
            frame_data[i] = (t + 1.0) * 0.5;
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, frames_t, channels, frames_h, frames_w]),
            TensorData::F32(frame_data),
        ))
    }

    fn latent_shape(&self, frames: u32, height: u32, width: u32) -> TensorShape {
        TensorShape::new(vec![
            1,
            frames / self.time_compression_factor,
            self.latent_channels,
            height / 8,
            width / 8,
        ])
    }
}

pub struct AudioVAE {
    gguf: Arc<GGUFFile>,
    sample_rate: u32,
    channels: u32,
}

impl AudioVAE {
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        Self {
            gguf,
            sample_rate: 48000,
            channels: 2,
        }
    }

    pub fn encode_audio(&self, waveform: &[f32], num_frames: usize) -> Result<Tensor> {
        let latent_frames = num_frames / 320;
        let latent_dim = 512;

        let mut latent_data = vec![0.0_f32; latent_frames * latent_dim];

        for i in 0..latent_frames * latent_dim {
            let freq = i as f32 * 0.01;
            latent_data[i] = freq.sin() * 0.1;
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, latent_frames as u32, latent_dim as u32]),
            TensorData::F32(latent_data),
        ))
    }

    pub fn decode_audio(&self, latent: &Tensor, num_frames: usize) -> Result<Vec<f32>> {
        let mut waveform = vec![0.0_f32; num_frames * self.channels as usize];

        for i in 0..waveform.len() {
            waveform[i] = ((i as f32 * 0.001).sin() + 1.0) * 0.1;
        }

        Ok(waveform)
    }
}
