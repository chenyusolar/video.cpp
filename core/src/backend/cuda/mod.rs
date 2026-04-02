#![allow(unused)]

use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Backend, Error, Result};
use std::sync::RwLock;

pub struct CudaBackend {
    device_id: usize,
    memory_allocated: RwLock<u64>,
    memory_reserved: RwLock<u64>,
}

impl CudaBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            tracing::info!("CUDA backend initializing for device {}", device_id);
            Ok(Self {
                device_id,
                memory_allocated: RwLock::new(0),
                memory_reserved: RwLock::new(0),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::Backend(
                "CUDA support not available. Compile with --features cuda".into(),
            ))
        }
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    fn update_memory(&self, bytes: u64) {
        *self.memory_allocated.write().unwrap() += bytes;
        *self.memory_reserved.write().unwrap() += bytes;
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn tensor_add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape = a.shape().to_vec();
        let data = match (a.data(), b.data()) {
            (TensorData::F32(a_data), TensorData::F32(b_data)) => {
                let mut result = Vec::with_capacity(a_data.len());
                for (x, y) in a_data.iter().zip(b_data.iter()) {
                    result.push(x + y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a_data), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a_data.len());
                for x in a_data.iter() {
                    result.push(x + s);
                }
                TensorData::F32(result)
            }
            _ => return Err(Error::Tensor("Unsupported types for add".into())),
        };
        let vol: u64 = shape.iter().map(|&x| x as u64).product();
        self.update_memory(vol * 4);
        Ok(Tensor::from_data(TensorShape::new(shape), data))
    }

    fn tensor_mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape = a.shape().to_vec();
        let data = match (a.data(), b.data()) {
            (TensorData::F32(a_data), TensorData::F32(b_data)) => {
                let mut result = Vec::with_capacity(a_data.len());
                for (x, y) in a_data.iter().zip(b_data.iter()) {
                    result.push(x * y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a_data), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a_data.len());
                for x in a_data.iter() {
                    result.push(x * s);
                }
                TensorData::F32(result)
            }
            _ => return Err(Error::Tensor("Unsupported types for mul".into())),
        };
        let vol: u64 = shape.iter().map(|&x| x as u64).product();
        self.update_memory(vol * 4);
        Ok(Tensor::from_data(TensorShape::new(shape), data))
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

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        self.update_memory((m * n) as u64 * 4);
        Ok(Tensor::from_data(
            TensorShape::new(vec![m as u32, n as u32]),
            TensorData::F32(result),
        ))
    }

    fn tensor_conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        opts: crate::libcore::tensor::Conv2dOpts,
    ) -> Result<Tensor> {
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

        for b in 0..batch {
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

                        output[b * out_channels * out_h * out_w
                            + oc * out_h * out_w
                            + oh * out_w
                            + ow] = sum;
                    }
                }
            }
        }

        self.update_memory((batch * out_channels * out_h * out_w) as u64 * 4);
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
        mask: Option<&Tensor>,
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

        self.update_memory(output.len() as u64 * 4);
        Ok(Tensor::from_data(
            TensorShape::new(vec![batch as u32, heads as u32, seq_len as u32, dim as u32]),
            TensorData::F32(output),
        ))
    }

    fn temporal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, _t: u32) -> Result<Tensor> {
        let q_shape = q.shape();
        let [B, T, H, W, C] = [q_shape[0], q_shape[1], q_shape[2], q_shape[3], q_shape[4]];

        let B = B as usize;
        let T = T as usize;
        let H = H as usize;
        let W = W as usize;
        let C = C as usize;

        let q_reshaped = q.reshape(&[(B * H * W) as u32, T as u32, C as u32])?;
        let k_reshaped = k.reshape(&[(B * H * W) as u32, T as u32, C as u32])?;
        let v_reshaped = v.reshape(&[(B * H * W) as u32, T as u32, C as u32])?;

        let result = self.attention(&q_reshaped, &k_reshaped, &v_reshaped, None)?;

        result.reshape(&[B as u32, T as u32, H as u32, W as u32, C as u32])
    }

    fn alloc_tensor(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size: u64 = shape.as_slice().iter().map(|&x| x as u64).product();
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![0.0_f32; size as usize]),
            DType::F16 => TensorData::F16(vec![half::f16::ZERO; size as usize]),
            DType::BF16 => TensorData::BF16(vec![half::bf16::ZERO; size as usize]),
            DType::I32 => TensorData::I32(vec![0_i32; size as usize]),
            DType::I64 => TensorData::I64(vec![0_i64; size as usize]),
            DType::U8 => TensorData::U8(vec![0_u8; size as usize]),
        };
        self.update_memory(size * dtype.size_of() as u64);
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

        let size: u64 = shape.as_slice().iter().map(|&x| x as u64).product();
        let mut rng = rand::thread_rng();

        let data: Vec<f32> = (0..size as usize)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        self.update_memory(size * 4);
        Ok(Tensor::from_data(shape, TensorData::F32(data)))
    }

    fn randn_like(&self, tensor: &Tensor) -> Result<Tensor> {
        self.randn(tensor.shape().clone())
    }

    fn zeros(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        self.alloc_tensor(shape, dtype)
    }

    fn ones(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size: u64 = shape.as_slice().iter().map(|&x| x as u64).product();
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![1.0_f32; size as usize]),
            DType::F16 => TensorData::F16(vec![half::f16::ONE; size as usize]),
            DType::BF16 => TensorData::BF16(vec![half::bf16::ONE; size as usize]),
            DType::I32 => TensorData::I32(vec![1_i32; size as usize]),
            DType::I64 => TensorData::I64(vec![1_i64; size as usize]),
            DType::U8 => TensorData::U8(vec![1_u8; size as usize]),
        };
        self.update_memory(size * dtype.size_of() as u64);
        Ok(Tensor::from_data(shape, data))
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn memory_allocated(&self) -> u64 {
        *self.memory_allocated.read().unwrap()
    }

    fn memory_reserved(&self) -> u64 {
        *self.memory_reserved.read().unwrap()
    }
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}
