use crate::libcore::tensor::Tensor;
use crate::libcore::traits::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub enum QuantType {
    None,
    Int8,
    NF4,
    FP8,
}

pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<u32>,
    pub dtype: QuantType,
    pub scale: Option<Vec<f32>>,
    pub zero_point: Option<Vec<u8>>,
}

impl QuantizedTensor {
    pub fn quantize_int8(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape().to_vec();
        let data = match tensor.data() {
            crate::libcore::tensor::TensorData::F32(arr) => {
                let mut quantized = Vec::with_capacity(arr.len());
                let mut scales = Vec::new();

                for chunk in arr.chunks(32) {
                    let max_val = chunk.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));
                    let scale = if max_val > 0.0 { 127.0 / max_val } else { 1.0 };

                    for &val in chunk {
                        let q = ((val * scale).round() as i32).clamp(-128, 127);
                        quantized.push(q as u8);
                    }
                    scales.push(1.0 / scale);
                }

                quantized
            }
            _ => {
                return Err(Error::Tensor(
                    "Only F32 tensors can be quantized to int8".into(),
                ))
            }
        };

        Ok(Self {
            data,
            shape,
            dtype: QuantType::Int8,
            scale: None,
            zero_point: None,
        })
    }

    pub fn quantize_nf4(tensor: &Tensor) -> Result<Self> {
        let shape = tensor.shape().to_vec();
        let data = match tensor.data() {
            crate::libcore::tensor::TensorData::F32(arr) => {
                let mut quantized = Vec::with_capacity((arr.len() + 1) / 2);
                let mut scales = Vec::new();

                let nf4_lookup = [
                    -1.0,
                    -0.6961928009986877,
                    -0.4132863298511505,
                    -0.23036947856903076,
                    0.0,
                    0.23036947856903076,
                    0.4132863298511505,
                    0.6961928009986877,
                    1.0,
                ];

                for chunk in arr.chunks(32) {
                    let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));
                    let scale = if max_abs > 0.0 { 1.0 / max_abs } else { 1.0 };

                    for pair in chunk.chunks(2) {
                        let mut q0 = 0u8;
                        let mut q1 = 0u8;

                        let v0 = pair[0] * scale;
                        let mut best_err0 = f32::MAX;
                        for (i, &lookup) in nf4_lookup.iter().enumerate() {
                            let err = (v0 - lookup).abs();
                            if err < best_err0 {
                                best_err0 = err;
                                q0 = i as u8;
                            }
                        }

                        if pair.len() > 1 {
                            let v1 = pair[1] * scale;
                            let mut best_err1 = f32::MAX;
                            for (i, &lookup) in nf4_lookup.iter().enumerate() {
                                let err = (v1 - lookup).abs();
                                if err < best_err1 {
                                    best_err1 = err;
                                    q1 = i as u8;
                                }
                            }
                        }

                        quantized.push((q1 << 4) | q0);
                    }

                    scales.push(scale);
                }

                quantized
            }
            _ => {
                return Err(Error::Tensor(
                    "Only F32 tensors can be quantized to NF4".into(),
                ))
            }
        };

        Ok(Self {
            data,
            shape,
            dtype: QuantType::NF4,
            scale: None,
            zero_point: None,
        })
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match self.dtype {
            QuantType::Int8 => self.dequantize_int8(),
            QuantType::NF4 => self.dequantize_nf4(),
            QuantType::FP8 => self.dequantize_fp8(),
            QuantType::None => Err(Error::Tensor("Cannot dequantize None type".into())),
        }
    }

    fn dequantize_int8(&self) -> Result<Tensor> {
        let mut data =
            Vec::with_capacity(self.shape.iter().map(|&d| d as usize).product::<usize>());

        for (i, &q) in self.data.iter().enumerate() {
            let scale = 1.0;
            let val = (q as i8 as f32) * scale;
            data.push(val);
        }

        Ok(Tensor::from_data(
            crate::libcore::tensor::TensorShape::new(self.shape.clone()),
            crate::libcore::tensor::TensorData::F32(data),
        ))
    }

    fn dequantize_nf4(&self) -> Result<Tensor> {
        let nf4_lookup = [
            -1.0,
            -0.6961928009986877,
            -0.4132863298511505,
            -0.23036947856903076,
            0.0,
            0.23036947856903076,
            0.4132863298511505,
            0.6961928009986877,
            1.0,
        ];

        let mut data =
            Vec::with_capacity(self.shape.iter().map(|&d| d as usize).product::<usize>());

        for &packed in &self.data {
            let q0 = (packed & 0x0F) as usize;
            let q1 = ((packed >> 4) & 0x0F) as usize;

            if q0 < nf4_lookup.len() {
                data.push(nf4_lookup[q0]);
            }
            if q1 < nf4_lookup.len() {
                data.push(nf4_lookup[q1]);
            }
        }

        Ok(Tensor::from_data(
            crate::libcore::tensor::TensorShape::new(self.shape.clone()),
            crate::libcore::tensor::TensorData::F32(data),
        ))
    }

    fn dequantize_fp8(&self) -> Result<Tensor> {
        Err(Error::Unsupported(
            "FP8 dequantization not yet implemented".into(),
        ))
    }
}
