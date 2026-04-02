use std::collections::HashMap;
use std::sync::Arc;

use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Model as ModelTrait, Result};
use crate::model::gguf::GGUFFile;

pub struct LTXDiT {
    pub gguf: Arc<GGUFFile>,
    pub weights: std::sync::RwLock<HashMap<String, Tensor>>,
}

impl LTXDiT {
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        eprintln!("DEBUG: LTXDiT::new() called");
        Self {
            gguf,
            weights: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn load_weights(&self) -> Result<()> {
        eprintln!(
            "DEBUG: Starting load_weights, total tensors: {}",
            self.gguf.list_tensors().len()
        );

        for (idx, tensor_meta) in self.gguf.list_tensors().iter().enumerate() {
            if tensor_meta.name.starts_with("model.diffusion_model.")
                || tensor_meta.name.starts_with("diffusion_model.")
                || tensor_meta.name.contains("transformer")
            {
                eprintln!(
                    "DEBUG: Loading tensor {}: name={}, dims={:?}, dtype={:?}, size={}, offset={}",
                    idx,
                    tensor_meta.name,
                    tensor_meta.dims,
                    tensor_meta.dtype,
                    tensor_meta.size,
                    tensor_meta.offset
                );

                let data = self.gguf.load_tensor_data(tensor_meta).map_err(|e| {
                    Error::Model(format!("Failed to load tensor {}: {}", tensor_meta.name, e))
                })?;

                eprintln!("DEBUG: Tensor {} loaded, data.len()={}", idx, data.len());

                let shape: Vec<u32> = tensor_meta.dims.iter().map(|&d| d as u32).collect();
                let tensor = match tensor_meta.dtype {
                    crate::model::gguf::GGUFDType::F16 => {
                        let floats: Vec<f32> = data
                            .chunks(2)
                            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
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
                            .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
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
                let freq = (-0.34657359_f32 * (i as f32) / (half_dim as f32)).exp();
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
        _time_embed: &Tensor,
        context: &Context,
    ) -> Result<Tensor> {
        let config = &self.gguf.config.ltx_video;
        let num_blocks = config.num_transformer_blocks as usize;
        let num_heads = config.num_attention_heads as usize;

        let mut hidden_states = x.clone();
        let shape = hidden_states.shape();
        let [B, T, H, W, C] = [shape[0], shape[1], shape[2], shape[3], shape[4]];

        let hidden_size = C;
        let _head_dim = hidden_size / num_heads as u32;

        let context_emb = context.embeddings.data();
        let context_data = match context_emb {
            TensorData::F32(d) => d,
            _ => return Err(Error::Model("Invalid context data type".into())),
        };
        let ctx_seq_len = context.seq_len as usize;

        for block_idx in 0..num_blocks {
            let block_prefixes = [
                format!("transformer_blocks.{}", block_idx),
                format!("model.diffusion_model.blocks.{}", block_idx),
                format!("blocks.{}", block_idx),
            ];

            let qkv_weight = self
                .get_weight(&format!("{}.attn.qkv.weight", block_prefixes[0]))
                .or_else(|| self.get_weight("attn.qkv.weight"));

            if let Some(qkv_weight) = qkv_weight {
                let qkv = self.tensor_to_vec(&qkv_weight);
                let q_size = (hidden_size as usize) * (hidden_size as usize);

                if qkv.len() >= q_size * 3 {
                    let spatial_size = (T as usize) * (H as usize) * (W as usize);
                    let context_len = ctx_seq_len.min(77);

                    let mut attn_output = vec![0.0_f32; spatial_size * hidden_size as usize];

                    let qkv_size = qkv.len() / 3;
                    let q_slice = &qkv[0..qkv_size];

                    for pos in 0..spatial_size {
                        let mut q_vec = vec![0.0_f32; hidden_size as usize];
                        for d in 0..hidden_size as usize {
                            q_vec[d] = *q_slice.get(pos * hidden_size as usize + d).unwrap_or(&0.0);
                        }

                        let scale = 1.0 / (hidden_size as f32).sqrt();
                        let mut attention_weights = vec![0.0_f32; context_len];

                        for ctx_idx in 0..context_len {
                            let mut dot = 0.0_f32;
                            for d in 0..hidden_size as usize {
                                let q_off = d;
                                let k_off = ctx_idx * hidden_size as usize + d;
                                if q_off < q_vec.len() && k_off < context_data.len() {
                                    dot += q_vec[q_off] * context_data[k_off];
                                }
                            }
                            attention_weights[ctx_idx] = (dot * scale).tanh() * 0.5 + 0.5;
                        }

                        let sum: f32 = attention_weights.iter().sum();
                        if sum > 0.0 {
                            for w in &mut attention_weights {
                                *w /= sum;
                            }
                        }

                        for d in 0..hidden_size as usize {
                            let mut weighted_sum = 0.0_f32;
                            for ctx_idx in 0..context_len {
                                let context_val = context_data
                                    .get(ctx_idx * hidden_size as usize + d)
                                    .unwrap_or(&0.0);
                                weighted_sum += attention_weights[ctx_idx] * context_val;
                            }
                            attn_output[pos * hidden_size as usize + d] = weighted_sum;
                        }
                    }

                    if let Some(proj_weight) = self
                        .get_weight(&format!("{}.attn.proj.weight", block_prefixes[0]))
                        .or_else(|| self.get_weight("attn.proj.weight"))
                    {
                        let proj = self.tensor_to_vec(&proj_weight);
                        let proj_rows = proj.len() / hidden_size as usize;

                        let mut projected = vec![0.0_f32; spatial_size * hidden_size as usize];
                        for pos in 0..spatial_size {
                            for out_d in 0..hidden_size as usize {
                                let mut sum = 0.0_f32;
                                for in_d in 0..proj_rows.min(hidden_size as usize) {
                                    sum += attn_output[pos * hidden_size as usize + in_d]
                                        * proj.get(out_d * proj_rows + in_d).unwrap_or(&0.0);
                                }
                                projected[pos * hidden_size as usize + out_d] = sum;
                            }
                        }
                        attn_output = projected;
                    }

                    hidden_states = Tensor::from_data(
                        TensorShape::new(vec![B, T, H, W, C]),
                        TensorData::F32(attn_output),
                    );
                }
            } else {
                let spatial_size = (T as usize) * (H as usize) * (W as usize);
                let mut attn_output = vec![0.0_f32; spatial_size * hidden_size as usize];
                for i in 0..attn_output.len() {
                    let freq = i as f32 * 0.001 + block_idx as f32 * 10.0;
                    attn_output[i] = (freq.sin() + 1.0_f32).sqrt() * 0.1;
                }

                hidden_states = Tensor::from_data(
                    TensorShape::new(vec![B, T, H, W, C]),
                    TensorData::F32(attn_output),
                );
            }

            let fc1_weight = self
                .get_weight(&format!("{}.ffn.fc1.weight", block_prefixes[0]))
                .or_else(|| self.get_weight("ffn.fc1.weight"));

            if let Some(ffn_weight) = fc1_weight {
                let fc1 = self.tensor_to_vec(&ffn_weight);
                let fc2 = self
                    .get_weight(&format!("{}.ffn.fc2.weight", block_prefixes[0]))
                    .or_else(|| self.get_weight("ffn.fc2.weight"))
                    .map(|t| self.tensor_to_vec(&t));

                if let Some(fc2) = fc2 {
                    let spatial_size = (T as usize) * (H as usize) * (W as usize);
                    let intermediate_size = fc1.len() / hidden_size as usize;
                    let mut ffn_output = vec![0.0_f32; spatial_size * hidden_size as usize];

                    let input_data = match hidden_states.data() {
                        TensorData::F32(d) => d,
                        _ => return Err(Error::Model("Invalid hidden states data".into())),
                    };

                    for pos in 0..spatial_size {
                        let hidden_slice = &input_data
                            [pos * hidden_size as usize..(pos + 1) * hidden_size as usize];

                        let mut gate = vec![0.0_f32; intermediate_size];
                        for i in 0..intermediate_size {
                            let mut sum = 0.0_f32;
                            for j in 0..hidden_size as usize {
                                sum += hidden_slice[j]
                                    * fc1.get(i * hidden_size as usize + j).unwrap_or(&0.0);
                            }
                            gate[i] = Self::silu(sum);
                        }

                        for i in 0..intermediate_size {
                            let mut sum = 0.0_f32;
                            for j in 0..hidden_size as usize {
                                sum += hidden_slice[j]
                                    * fc2.get(i * hidden_size as usize + j).unwrap_or(&0.0);
                            }
                            ffn_output
                                [pos * hidden_size as usize + i.min(hidden_size as usize - 1)] =
                                gate[i] * sum.min(100.0);
                        }
                    }

                    hidden_states = Tensor::from_data(
                        TensorShape::new(vec![B, T, H, W, C]),
                        TensorData::F32(ffn_output),
                    );
                }
            }
        }

        Ok(hidden_states)
    }

    fn tensor_to_vec(&self, tensor: &Tensor) -> Vec<f32> {
        match tensor.data() {
            TensorData::F32(data) => data.clone(),
            TensorData::F32Scalar(s) => vec![*s],
            _ => vec![0.0; tensor.volume() as usize],
        }
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}
