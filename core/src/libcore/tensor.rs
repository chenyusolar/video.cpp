use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
        }
    }
}

#[derive(Debug, Clone, Clone)]
pub struct TensorShape {
    dims: Vec<u32>,
}

impl TensorShape {
    pub fn new(dims: Vec<u32>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims
    }

    pub fn volume(&self) -> u32 {
        self.dims.iter().product()
    }

    pub fn len(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&u32> {
        self.dims.get(i)
    }

    pub fn reshape(&self, dims: Vec<u32>) -> Self {
        Self { dims }
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F32Scalar(f32),
    F16(Vec<f16>),
    BF16(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            TensorData::F32(_) => DType::F32,
            TensorData::F32Scalar(_) => DType::F32,
            TensorData::F16(_) => DType::F16,
            TensorData::BF16(_) => DType::BF16,
            TensorData::I32(_) => DType::I32,
            TensorData::I64(_) => DType::I64,
            TensorData::U8(_) => DType::U8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: TensorShape,
    data: TensorData,
    device: Device,
}

impl Tensor {
    pub fn from_data(shape: TensorShape, data: TensorData) -> Self {
        Self {
            shape,
            data,
            device: Device::CPU,
        }
    }

    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    pub fn data(&self) -> &TensorData {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut TensorData {
        &mut self.data
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn volume(&self) -> u32 {
        self.shape.volume()
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, super::super::Error> {
        use super::super::traits::Result;

        let shape = self.shape.clone();
        let result_data = match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let mut result = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    result.push(x + y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a.len());
                for x in a.iter() {
                    result.push(x + s);
                }
                TensorData::F32(result)
            }
            _ => return Err(super::super::Error::Tensor("Unsupported add types".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, super::super::Error> {
        use super::super::traits::Result;

        let shape = self.shape.clone();
        let result_data = match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let mut result = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    result.push(x - y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a.len());
                for x in a.iter() {
                    result.push(x - s);
                }
                TensorData::F32(result)
            }
            _ => return Err(super::super::Error::Tensor("Unsupported sub types".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, super::super::Error> {
        use super::super::traits::Result;

        let shape = self.shape.clone();
        let result_data = match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let mut result = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    result.push(x * y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a.len());
                for x in a.iter() {
                    result.push(x * s);
                }
                TensorData::F32(result)
            }
            _ => return Err(super::super::Error::Tensor("Unsupported mul types".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, super::super::Error> {
        use super::super::traits::Result;

        let shape = self.shape.clone();
        let result_data = match (&self.data, &other.data) {
            (TensorData::F32(a), TensorData::F32(b)) => {
                let mut result = Vec::with_capacity(a.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    result.push(x / y);
                }
                TensorData::F32(result)
            }
            (TensorData::F32(a), TensorData::F32Scalar(s)) => {
                let mut result = Vec::with_capacity(a.len());
                for x in a.iter() {
                    result.push(x / s);
                }
                TensorData::F32(result)
            }
            _ => return Err(super::super::Error::Tensor("Unsupported div types".into())),
        };
        Ok(Tensor::from_data(shape, result_data))
    }

    pub fn reshape(&self, dims: &[u32]) -> Result<Tensor, super::super::Error> {
        let new_volume: u32 = dims.iter().product();
        if new_volume != self.volume() {
            return Err(super::super::Error::Tensor(
                format!(
                    "Cannot reshape {} to {} - volume mismatch",
                    self.volume(),
                    new_volume
                )
                .into(),
            ));
        }
        Ok(Tensor::from_data(
            TensorShape::new(dims.to_vec()),
            self.data.clone(),
        ))
    }

    pub fn randn_like(&self) -> Result<Tensor, super::super::Error> {
        use super::super::traits::Result;
        use rand::prelude::*;

        let size = self.volume() as usize;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        Ok(Tensor::from_data(self.shape.clone(), TensorData::F32(data)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize),
    Vulkan,
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::CUDA(id) => write!(f, "CUDA:{}", id),
            Device::Vulkan => write!(f, "Vulkan"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Conv2dOpts {
    pub stride: Option<usize>,
    pub padding: Option<usize>,
    pub dilation: Option<usize>,
    pub groups: Option<usize>,
}

impl Default for Conv2dOpts {
    fn default() -> Self {
        Self {
            stride: Some(1),
            padding: Some(0),
            dilation: Some(1),
            groups: Some(1),
        }
    }
}
