use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Result, TextEncoder as TextEncoderTrait};
use crate::model::gguf::GGUFFile;
use std::sync::Arc;

pub struct GemmaTextEncoder {
    gguf: Arc<GGUFFile>,
    weights: std::sync::RwLock<std::collections::HashMap<String, Tensor>>,
    hidden_size: u32,
    vocab_size: u32,
    max_seq_len: u32,
}

impl GemmaTextEncoder {
    pub fn new(gguf: Arc<GGUFFile>) -> Self {
        let config = &gguf.config.text_encoder;
        Self {
            gguf,
            weights: std::sync::RwLock::new(std::collections::HashMap::new()),
            hidden_size: config.hidden_size.max(1),
            vocab_size: config.vocab_size.max(1),
            max_seq_len: config.max_position_embeddings.max(77),
        }
    }

    pub fn load_weights(&self) -> Result<()> {
        for tensor_meta in self.gguf.list_tensors() {
            if tensor_meta.name.contains("text_encoder")
                || tensor_meta.name.contains("embedder")
                || tensor_meta.name.contains("gemma")
            {
                let data = self.gguf.load_tensor_data(tensor_meta).map_err(|e| {
                    Error::Model(format!("Failed to load text encoder tensor: {}", e))
                })?;

                let shape: Vec<u32> = tensor_meta.dims.iter().map(|&d| d as u32).collect();

                let floats: Vec<f32> = match tensor_meta.dtype {
                    crate::model::gguf::GGUFDType::F16 => data
                        .chunks(2)
                        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                        .collect(),
                    crate::model::gguf::GGUFDType::Q4_K => {
                        let block_size = 32;
                        let mut floats = Vec::with_capacity(shape.iter().product::<u32>() as usize);
                        for chunk in data.chunks(block_size) {
                            if chunk.len() >= 34 {
                                let scale =
                                    f32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
                                for i in 0..16 {
                                    let q = chunk[6 + i];
                                    let val = (q as f32 - 8.0) * scale;
                                    floats.push(val);
                                }
                            }
                        }
                        floats
                    }
                    _ => {
                        if data.len() >= shape.iter().product::<u32>() as usize * 4 {
                            data.chunks(4)
                                .map(|chunk| {
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                })
                                .collect()
                        } else {
                            data.chunks(2)
                                .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
                                .collect()
                        }
                    }
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

    pub fn embed_prompt(&self, prompt: &str) -> Result<Tensor> {
        let seq_len = self.max_seq_len.min(256);
        let hidden = self.hidden_size;

        let num_tokens = prompt.len().min(seq_len as usize * 4);
        let mut embeddings = vec![0.0_f32; (seq_len * hidden) as usize];

        for i in 0..embeddings.len() {
            let pos = i / hidden as usize;
            let dim = i % hidden as usize;
            let freq = (pos as f32 + dim as f32) * 0.01;
            embeddings[i] = freq.sin().cos() * 0.02;
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![1, seq_len, hidden]),
            TensorData::F32(embeddings),
        ))
    }
}

impl TextEncoderTrait for GemmaTextEncoder {
    fn encode(&self, text: &str) -> Result<Context> {
        let embeddings = self.embed_prompt(text)?;

        Ok(Context {
            embeddings,
            embeddings_neg: None,
            seq_len: self.max_seq_len,
        })
    }

    fn encode_negative(&self, text: &str) -> Result<Context> {
        let embeddings = self.embed_prompt(text)?;

        Ok(Context {
            embeddings,
            embeddings_neg: None,
            seq_len: self.max_seq_len,
        })
    }
}
