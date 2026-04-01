use rayon::prelude::*;
use std::sync::Arc;

use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Model as ModelTrait, Result};
use crate::model::gguf::GGUFFile;

pub struct LTXDiT {
    pub gguf: Arc<GGUFFile>,
    pub weights: std::sync::RwLock<HashMap<String, Tensor>>,
}

use std::collections::HashMap;

impl LTXDiT {
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        Self {
            gguf,
            weights: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn load_weights(&self) -> Result<()> {
        for tensor_meta in self.gguf.list_tensors() {
            if tensor_meta.name.starts_with("model.diffusion_model.")
                || tensor_meta.name.starts_with("diffusion_model.")
                || tensor_meta.name.contains("transformer")
            {
                let data = self.gguf.load_tensor_data(tensor_meta).map_err(|e| {
                    Error::Model(format!("Failed to load tensor {}: {}", tensor_meta.name, e))
                })?;

                let shape: Vec<u32> = tensor_meta.dims.iter().map(|&d| d as u32).collect();
                let tensor = match tensor_meta.dtype {
                    crate::model::gguf::GGUFDType::F16 => {
                        let floats: Vec<f32> = data
                            .chunks(2)
                            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                            .collect();
                        Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats))
                    }
                    crate::model::gguf::GGUFDType::BF16 => {
                        let floats: Vec<f32> = data
                            .chunks(2)
                            .map(|chunk| {
                                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                f32::from_bits((bits as u32) << 16)
                            })
                            .collect();
                        Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats))
                    }
                    crate::model::gguf::GGUFDType::F32 => {
                        let floats: Vec<f32> = data
                            .chunks(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats))
                    }
                    _ => {
                        let floats: Vec<f32> = data
                            .chunks(2)
                            .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                            .collect();
                        Tensor::from_data(TensorShape::new(shape), TensorData::F32(floats))
                    }
                };

                self.weights
                    .write()
                    .unwrap()
                    .insert(tensor_meta.name.clone(), tensor);
            }
        }
        Ok(())
    }

    pub fn get_weight(&self, name: &str) -> Option<Tensor> {
        self.weights.read().unwrap().get(name).cloned()
    }

    pub fn parameters(&self) -> usize {
        self.weights
            .read()
            .unwrap()
            .values()
            .map(|t| t.volume() as usize)
            .sum()
    }
}

impl ModelTrait for LTXDiT {
    fn forward(&self, latent: &Tensor, timestep: f32, context: &Context) -> Result<Tensor> {
        let config = &self.gguf.config.ltx_video;

        let time_embed = self.get_timestep_embedding(timestep, config.time_embed_dim)?;

        let mut hidden_states = latent.clone();

        hidden_states = self.apply_transformer_blocks(&hidden_states, &time_embed, context)?;

        Ok(hidden_states)
    }

    fn parameters(&self) -> usize {
        self.parameters()
    }
}

impl LTXDiT {
    fn get_timestep_embedding(&self, timestep: f32, dim: u32) -> Result<Tensor> {
        let half_dim = dim / 2;

        let emb: Vec<f32> = (0..half_dim)
            .flat_map(|i| {
                let freq = (-std::f32::consts::LOG_SQRT_2 * (i as f32) / (half_dim as f32)).exp();
                let arg = timestep * freq * std::f32::consts::PI;
                vec![arg.sin(), arg.cos()]
            })
            .collect();

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, dim]),
            TensorData::F32(emb),
        ))
    }

    fn apply_transformer_blocks(
        &self,
        x: &Tensor,
        time_embed: &Tensor,
        context: &Context,
    ) -> Result<Tensor> {
        let x = x.clone();
        Ok(x)
    }
}
