use rayon::prelude::*;
use std::sync::Arc;

use crate::libcore::tensor::Conv2dOpts;
use crate::libcore::tensor::{DType, Device, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Backend, Error, Result};

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        let num_threads = rayon::current_num_threads();
        Self { num_threads }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn tensor_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape = a.shape();
        let result_data = match (a.data(), b.data()) {
            (TensorData::F32(a_data), TensorData::F32(b_data)) => {
                let mut result = Vec::with_capacity(a_data.len());
                result.par_extend(a_data.par_iter().zip(b_data.par_iter()).map(|(x, y)| x + y));
                TensorData::F32(result)
            }
            _ => return Err(Error::Tensor("Unsupported data types for add".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    fn tensor_mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape = a.shape();
        let result_data = match (a.data(), b.data()) {
            (TensorData::F32(a_data), TensorData::F32(b_data)) => {
                let mut result = Vec::with_capacity(a_data.len());
                result.par_extend(a_data.par_iter().zip(b_data.par_iter()).map(|(x, y)| x * y));
                TensorData::F32(result)
            }
            (TensorData::F32(a_data), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a_data.len());
                result.par_extend(a_data.par_iter().map(|x| x * *s));
                TensorData::F32(result)
            }
            _ => return Err(Error::Tensor("Unsupported data types for mul".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(Error::Tensor("matmul requires 2D tensors".into()));
        }

        let (m, k) = (a_shape[0] as usize, a_shape[1] as usize);
        let (k2, n) = (b_shape[0] as usize, b_shape[1] as usize);

        if k != k2 {
            return Err(Error::Tensor(format!(
                "matmul dimension mismatch: {}x{} @ {}x{}",
                m, k, k2, n
            )));
        }

        let a_data = match a.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for matmul".into())),
        };
        let b_data = match b.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for matmul".into())),
        };

        let mut result = vec![0.0_f32; m * n];

        result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                row[j] = sum;
            }
        });

        Ok(Tensor::from_data(
            TensorShape::new(vec![m as u32, n as u32]),
            TensorData::F32(result),
        ))
    }

    fn tensor_conv2d(&self, input: &Tensor, weight: &Tensor, opts: Conv2dOpts) -> Result<Tensor> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();

        let batch = input_shape[0] as usize;
        let in_channels = input_shape[1] as usize;
        let in_h = input_shape[2] as usize;
        let in_w = input_shape[3] as usize;

        let out_channels = weight_shape[0] as usize;
        let k_h = weight_shape[2] as usize;
        let k_w = weight_shape[3] as usize;

        let stride = opts.stride.unwrap_or(1);
        let padding = opts.padding.unwrap_or(0);
        let dilation = opts.dilation.unwrap_or(1);

        let out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let input_data = match input.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for conv2d".into())),
        };
        let weight_data = match weight.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for conv2d".into())),
        };

        let mut output = vec![0.0_f32; batch * out_channels * out_h * out_w];

        output
            .par_chunks_mut(out_channels * out_h * out_w)
            .enumerate()
            .for_each(|(b, out_batch)| {
                for oc in 0..out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0_f32;

                            for ic in 0..in_channels {
                                for kh in 0..k_h {
                                    for kw in 0..k_w {
                                        let ih = oh * stride + kh * dilation - padding;
                                        let iw = ow * stride + kw * dilation - padding;

                                        if ih < in_h && iw < in_w {
                                            let input_idx = b * in_channels * in_h * in_w
                                                + ic * in_h * in_w
                                                + ih * in_w
                                                + iw;
                                            let weight_idx = oc * in_channels * k_h * k_w
                                                + ic * k_h * k_w
                                                + kh * k_w
                                                + kw;
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }

                            out_batch[oc * out_h * out_w + oh * out_w + ow] = sum;
                        }
                    }
                }
            });

        Ok(Tensor::from_data(
            TensorShape::new(vec![
                batch as u32,
                out_channels as u32,
                out_h as u32,
                out_w as u32,
            ]),
            TensorData::F32(output),
        ))
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        let batch = q_shape[0] as usize;
        let heads = q_shape[1] as usize;
        let seq_len = q_shape[2] as usize;
        let dim = q_shape[3] as usize;

        let scale = 1.0 / (dim as f32).sqrt();

        let q_data = match q.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for attention".into())),
        };
        let k_data = match k.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for attention".into())),
        };
        let v_data = match v.data() {
            TensorData::F32(d) => d,
            _ => return Err(Error::Tensor("Unsupported dtype for attention".into())),
        };

        let mut output = vec![0.0_f32; batch * heads * seq_len * dim];
        let mut attention_weights = vec![0.0_f32; batch * heads * seq_len * seq_len];

        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq_len {
                    let mut max_score = f32::MIN;

                    for j in 0..seq_len {
                        let mut score = 0.0_f32;
                        for d in 0..dim {
                            let q_idx = b * heads * seq_len * dim + h * seq_len * dim + i * dim + d;
                            let k_idx = b * heads * seq_len * dim + h * seq_len * dim + j * dim + d;
                            score += q_data[q_idx] * k_data[k_idx];
                        }
                        score *= scale;

                        let attn_idx =
                            b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        attention_weights[attn_idx] = score;

                        if score > max_score {
                            max_score = score;
                        }
                    }

                    for j in 0..seq_len {
                        let attn_idx =
                            b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        attention_weights[attn_idx] =
                            (attention_weights[attn_idx] - max_score).exp();
                    }

                    let mut sum = 0.0_f32;
                    for j in 0..seq_len {
                        let attn_idx =
                            b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        sum += attention_weights[attn_idx];
                    }

                    for j in 0..seq_len {
                        let attn_idx =
                            b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        attention_weights[attn_idx] /= sum;
                    }

                    for d in 0..dim {
                        let mut val = 0.0_f32;
                        for j in 0..seq_len {
                            let attn_idx = b * heads * seq_len * seq_len
                                + h * seq_len * seq_len
                                + i * seq_len
                                + j;
                            let v_idx = b * heads * seq_len * dim + h * seq_len * dim + j * dim + d;
                            val += attention_weights[attn_idx] * v_data[v_idx];
                        }
                        let out_idx = b * heads * seq_len * dim + h * seq_len * dim + i * dim + d;
                        output[out_idx] = val;
                    }
                }
            }
        }

        Ok(Tensor::from_data(
            TensorShape::new(vec![batch as u32, heads as u32, seq_len as u32, dim as u32]),
            TensorData::F32(output),
        ))
    }

    fn temporal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, t: u32) -> Result<Tensor> {
        let q_shape = q.shape();
        let [B, T, H, W, C] = q_shape else {
            return Err(Error::Tensor(
                "temporal_attention requires 5D tensor [B,T,H,W,C]".into(),
            ));
        };

        let B = B as usize;
        let T = T as usize;
        let H = H as usize;
        let W = W as usize;
        let C = C as usize;

        let q_reshaped = self.reshape(q, &[B * H * W, T, C])?;
        let k_reshaped = self.reshape(k, &[B * H * W, T, C])?;
        let v_reshaped = self.reshape(v, &[B * H * W, T, C])?;

        self.attention(&q_reshaped, &k_reshaped, &v_reshaped, None)?;

        self.reshape(&q_reshaped, &[B, T, H, W, C])
    }

    fn alloc_tensor(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size = shape.volume() as usize;
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![0.0_f32; size]),
            DType::F16 => TensorData::F16(vec![0.0_f16; size]),
            DType::BF16 => TensorData::BF16(vec![0.0_f32; size]),
            DType::I32 => TensorData::I32(vec![0_i32; size]),
            DType::I64 => TensorData::I64(vec![0_i64; size]),
            DType::U8 => TensorData::U8(vec![0_u8; size]),
        };
        Ok(Tensor::from_data(shape, data))
    }

    fn copy_to_device(&self, data: &[f32], tensor: &mut Tensor) -> Result<()> {
        match tensor.data_mut() {
            TensorData::F32(tensor_data) => {
                tensor_data.copy_from_slice(data);
                Ok(())
            }
            _ => Err(Error::Tensor("Unsupported dtype for copy_to_device".into())),
        }
    }

    fn copy_to_host(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor.data() {
            TensorData::F32(data) => Ok(data.clone()),
            TensorData::F32Scalar(s) => Ok(vec![*s]),
            _ => Err(Error::Tensor("Unsupported dtype for copy_to_host".into())),
        }
    }

    fn randn(&self, shape: TensorShape) -> Result<Tensor> {
        use rand::prelude::*;

        let size = shape.volume() as usize;
        let mut rng = rand::thread_rng();

        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::pi * u2).cos();
                normal
            })
            .collect();

        Ok(Tensor::from_data(shape, TensorData::F32(data)))
    }

    fn randn_like(&self, tensor: &Tensor) -> Result<Tensor> {
        self.randn(tensor.shape())
    }

    fn zeros(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size = shape.volume() as usize;
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![0.0_f32; size]),
            DType::F16 => TensorData::F16(vec![0.0_f16; size]),
            DType::BF16 => TensorData::BF16(vec![0.0_f32; size]),
            DType::I32 => TensorData::I32(vec![0_i32; size]),
            DType::I64 => TensorData::I64(vec![0_i64; size]),
            DType::U8 => TensorData::U8(vec![0_u8; size]),
        };
        Ok(Tensor::from_data(shape, data))
    }

    fn ones(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size = shape.volume() as usize;
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![1.0_f32; size]),
            DType::F16 => TensorData::F16(vec![1.0_f16; size]),
            DType::BF16 => TensorData::BF16(vec![1.0_f32; size]),
            DType::I32 => TensorData::I32(vec![1_i32; size]),
            DType::I64 => TensorData::I64(vec![1_i64; size]),
            DType::U8 => TensorData::U8(vec![1_u8; size]),
        };
        Ok(Tensor::from_data(shape, data))
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn memory_allocated(&self) -> u64 {
        0
    }

    fn memory_reserved(&self) -> u64 {
        0
    }
}
