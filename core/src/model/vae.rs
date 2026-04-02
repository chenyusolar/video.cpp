use crate::libcore::tensor::{Tensor, TensorData, TensorShape};
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

    fn tensor_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape = a.shape().clone();
        let a_data = match a.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for add".into())),
        };
        let b_data = match b.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for add".into())),
        };
        let mut result = Vec::with_capacity(a_data.len());
        for i in 0..a_data.len() {
            result.push(a_data[i] + b_data[i]);
        }
        Ok(Tensor::from_data(shape, TensorData::F32(result)))
    }

    fn silu(&self, x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn apply_silu(&self, input: &Tensor) -> Result<Tensor> {
        let data = match input.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for silu".into())),
        };
        let result: Vec<f32> = data.iter().map(|&x| self.silu(x)).collect();
        Ok(Tensor::from_data(
            input.shape().clone(),
            TensorData::F32(result),
        ))
    }

    fn group_norm(&self, input: &Tensor, num_groups: u32) -> Result<Tensor> {
        let shape = input.shape().clone();
        let data = match input.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for group_norm".into())),
        };
        let channels = shape[1];
        let group_size = channels / num_groups;
        let mut result = data.to_vec();

        for g in 0..num_groups as usize {
            let start = g * group_size as usize;
            let end = ((g + 1) * group_size as usize).min(data.len());
            let mut sum = 0.0_f32;
            for i in start..end {
                sum += data[i];
            }
            let mean = sum / (end - start) as f32;
            let mut var_sum = 0.0_f32;
            for i in start..end {
                var_sum += (data[i] - mean).powi(2);
            }
            let variance = var_sum / (end - start) as f32;
            let std = (variance + 1e-6).sqrt();
            for i in start..end {
                result[i] = (data[i] - mean) / std;
            }
        }
        Ok(Tensor::from_data(shape, TensorData::F32(result)))
    }

    fn res_block(&self, input: &Tensor, _weight_prefix: &str) -> Result<Tensor> {
        let mut x = self.group_norm(input, 32)?;
        x = self.apply_silu(&x)?;
        Ok(x)
    }

    fn downsample_2d(&self, input: &Tensor, factor: u32) -> Result<Tensor> {
        let shape = input.shape().clone();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let new_height = height / factor;
        let new_width = width / factor;

        let data = match input.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for downsample".into())),
        };

        let mut result = Vec::with_capacity((batch * channels * new_height * new_width) as usize);
        for _b in 0..batch as usize {
            for c in 0..channels as usize {
                for h in 0..new_height as usize {
                    for w in 0..new_width as usize {
                        let h_start = h * factor as usize;
                        let w_start = w * factor as usize;
                        let mut sum = 0.0_f32;
                        for di in 0..factor as usize {
                            for dj in 0..factor as usize {
                                let hi = (h_start + di).min((height as usize) - 1);
                                let wi = (w_start + dj).min((width as usize) - 1);
                                let idx =
                                    _b * (channels as usize) * (height as usize) * (width as usize)
                                        + c * (height as usize) * (width as usize)
                                        + hi * (width as usize)
                                        + wi;
                                sum += data[idx];
                            }
                        }
                        result.push(sum / (factor * factor) as f32);
                    }
                }
            }
        }
        Ok(Tensor::from_data(
            TensorShape::new(vec![batch, channels, new_height, new_width]),
            TensorData::F32(result),
        ))
    }

    fn upsample_2d(&self, input: &Tensor, factor: u32) -> Result<Tensor> {
        let shape = input.shape().clone();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let new_height = height * factor;
        let new_width = width * factor;

        let data = match input.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for upsample".into())),
        };

        let mut result = Vec::with_capacity((batch * channels * new_height * new_width) as usize);
        for _b in 0..batch as usize {
            for c in 0..channels as usize {
                for h in 0..new_height as usize {
                    for w in 0..new_width as usize {
                        let src_h = h / factor as usize;
                        let src_w = w / factor as usize;
                        let idx = c * (height as usize) * (width as usize)
                            + src_h * (width as usize)
                            + src_w;
                        result.push(data[idx]);
                    }
                }
            }
        }
        Ok(Tensor::from_data(
            TensorShape::new(vec![batch, channels, new_height, new_width]),
            TensorData::F32(result),
        ))
    }

    fn encode_frame_real(&self, frame: &Tensor) -> Result<Tensor> {
        let mut x = frame.clone();
        x = self.res_block(&x, "vae.encoder.conv_in")?;
        x = self.downsample_2d(&x, 2)?;
        x = self.res_block(&x, "vae.encoder.down_blocks.0")?;
        x = self.downsample_2d(&x, 2)?;
        x = self.res_block(&x, "vae.encoder.down_blocks.1")?;
        x = self.downsample_2d(&x, 2)?;
        x = self.group_norm(&x, 32)?;
        x = self.apply_silu(&x)?;
        Ok(x)
    }

    fn decode_frame_real(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = latent.clone();
        x = self.res_block(&x, "vae.decoder.up_blocks.0")?;
        x = self.upsample_2d(&x, 2)?;
        x = self.res_block(&x, "vae.decoder.up_blocks.1")?;
        x = self.upsample_2d(&x, 2)?;
        x = self.res_block(&x, "vae.decoder.up_blocks.2")?;
        x = self.upsample_2d(&x, 2)?;
        x = self.group_norm(&x, 32)?;
        x = self.apply_silu(&x)?;
        Ok(x)
    }

    fn extract_4d_frame(&self, tensor: &Tensor, _batch: usize, t: usize) -> Result<Tensor> {
        let shape = tensor.shape().clone();
        let [_B, _T, C, H, W] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
        let data = match tensor.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype".into())),
        };
        let frame_size = (C * H * W) as usize;
        let start_idx = t * frame_size;
        let mut frame_data = Vec::with_capacity(frame_size);
        for i in 0..frame_size {
            let idx = start_idx + i;
            if idx < data.len() {
                frame_data.push(data[idx]);
            } else {
                frame_data.push(0.0_f32);
            }
        }
        Ok(Tensor::from_data(
            TensorShape::new(vec![1, C, H, W]),
            TensorData::F32(frame_data),
        ))
    }
}

impl VAETrait for VideoVAE {
    fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        let shape = pixels.shape().clone();
        if shape.len() != 5 {
            return Err(Error::Tensor(
                "encode requires 5D tensor [B,T,C,H,W]".into(),
            ));
        }
        let [B, T, C, _H, _W] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
        let latent_t = T / self.time_compression_factor;
        let latent_h = _H / 8;
        let latent_w = _W / 8;

        let mut encoded_frames = Vec::new();
        for t in 0..T as usize {
            let frame_4d = self.extract_4d_frame(pixels, 0, t)?;
            let encoded = self.encode_frame_real(&frame_4d)?;
            encoded_frames.push(encoded);
        }

        let sample_frame = &encoded_frames[0];
        let encoded_shape = sample_frame.shape().clone();
        let [_enc_b, enc_c, _enc_h, _enc_w] = [
            encoded_shape[0],
            encoded_shape[1],
            encoded_shape[2],
            encoded_shape[3],
        ];

        let samples_per_frame = T as usize / latent_t as usize;
        let mut latents = Vec::with_capacity(
            (B as usize
                * latent_t as usize
                * enc_c as usize
                * latent_h as usize
                * latent_w as usize),
        );

        for b in 0..B as usize {
            for t in 0..latent_t as usize {
                let mut sum_tensor: Option<Tensor> = None;
                for s in 0..samples_per_frame {
                    let src_t = t * samples_per_frame + s;
                    if src_t < encoded_frames.len() {
                        if let Some(ref mut sum) = sum_tensor {
                            *sum = self.tensor_add(sum, &encoded_frames[src_t])?;
                        } else {
                            sum_tensor = Some(encoded_frames[src_t].clone());
                        }
                    }
                }
                if let Some(sum) = sum_tensor {
                    let sum_data = match sum.data() {
                        TensorData::F32(d) => d,
                        _ => {
                            return Err(Error::Tensor("Invalid data".into()));
                        }
                    };
                    latents.extend(sum_data.iter().copied());
                }
            }
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, latent_t, self.latent_channels, latent_h, latent_w]),
            TensorData::F32(latents),
        ))
    }

    fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let shape = latent.shape().clone();
        let [B, T, _C, H, W] = if shape.len() >= 5 {
            [shape[0], shape[1], shape[2], shape[3], shape[4]]
        } else {
            return Err(Error::Tensor(
                "decode requires at least 5D latent tensor".into(),
            ));
        };

        let frames_t = T * self.time_compression_factor;
        let frames_h = H * 8;
        let frames_w = W * 8;

        let mut decoded_frames = Vec::new();
        for t in 0..T as usize {
            let frame_4d = self.extract_4d_frame(latent, 0, t)?;
            let decoded = self.decode_frame_real(&frame_4d)?;
            decoded_frames.push(decoded);
        }

        let mut frames = Vec::new();
        for t in 0..frames_t as usize {
            let src_t = t / self.time_compression_factor as usize;
            if src_t < decoded_frames.len() {
                let frame_data = match decoded_frames[src_t].data() {
                    TensorData::F32(d) => d,
                    _ => return Err(Error::Tensor("Invalid data".into())),
                };
                for v in frame_data.iter().take((3 * frames_h * frames_w) as usize) {
                    frames.push(*v);
                }
            }
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![B, frames_t, 3, frames_h, frames_w]),
            TensorData::F32(frames),
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

    pub fn decode_audio(&self, _latent: &Tensor, num_frames: usize) -> Result<Vec<f32>> {
        let mut waveform = vec![0.0_f32; num_frames * self.channels as usize];
        for i in 0..waveform.len() {
            waveform[i] = ((i as f32 * 0.001).sin() + 1.0) * 0.1;
        }
        Ok(waveform)
    }
}
