#![allow(unused)]

use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Backend, Error, Result};

pub struct VulkanBackend {
    device_id: usize,
}

impl VulkanBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        {
            Ok(Self { device_id })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Err(Error::Backend("Vulkan support not compiled in".into()))
        }
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
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
            _ => return Err(Error::Tensor("Unsupported types".into())),
        };
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
            _ => return Err(Error::Tensor("Unsupported types".into())),
        };
        Ok(Tensor::from_data(TensorShape::new(shape), data))
    }

    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Vulkan matmul not yet implemented".into(),
        ))
    }

    fn tensor_conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        opts: crate::libcore::tensor::Conv2dOpts,
    ) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Vulkan conv2d not yet implemented".into(),
        ))
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Vulkan attention not yet implemented".into(),
        ))
    }

    fn temporal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, t: u32) -> Result<Tensor> {
        Err(Error::Unsupported(
            "Vulkan temporal_attention not yet implemented".into(),
        ))
    }

    fn alloc_tensor(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size = shape.volume() as usize;
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![0.0_f32; size]),
            DType::F16 => TensorData::F16(vec![half::f16::ZERO; size]),
            DType::BF16 => TensorData::BF16(vec![half::bf16::ZERO; size]),
            DType::I32 => TensorData::I32(vec![0_i32; size]),
            DType::I64 => TensorData::I64(vec![0_i64; size]),
            DType::U8 => TensorData::U8(vec![0_u8; size]),
        };
        Ok(Tensor::from_data(shape, data))
    }

    fn copy_to_device(&self, data: &[f32], tensor: &mut Tensor) -> Result<()> {
        Err(Error::Unsupported(
            "Vulkan copy_to_device not yet implemented".into(),
        ))
    }

    fn copy_to_host(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor.data() {
            TensorData::F32(data) => Ok(data.clone()),
            _ => Err(Error::Tensor("Unsupported type".into())),
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
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        Ok(Tensor::from_data(shape, TensorData::F32(data)))
    }

    fn randn_like(&self, tensor: &Tensor) -> Result<Tensor> {
        self.randn(tensor.shape().clone())
    }

    fn zeros(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        self.alloc_tensor(shape, dtype)
    }

    fn ones(&self, shape: TensorShape, dtype: DType) -> Result<Tensor> {
        let size = shape.volume() as usize;
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![1.0_f32; size]),
            DType::F16 => TensorData::F16(vec![half::f16::from_f32(1.0); size]),
            DType::BF16 => TensorData::BF16(vec![half::bf16::from_f32(1.0); size]),
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
