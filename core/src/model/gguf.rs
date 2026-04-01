use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GGUFConfig {
    pub general: GeneralConfig,
    pub ltx_video: LTXVideoConfig,
    pub text_encoder: TextEncoderConfig,
    pub vae: VAEConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub architecture: String,
    pub model_name: String,
    pub model_version: String,
    pub quantization_version: u32,
    pub alignment: u32,
    pub tensor_data_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXVideoConfig {
    pub latent_channels: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub time_embed_dim: u32,
    pub transform_layers: u32,
    pub num_attention_heads: u32,
    pub num_transformer_blocks: u32,
    pub frame_rate: u32,
    pub latent_height: u32,
    pub latent_width: u32,
    pub latent_frames: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEncoderConfig {
    pub model_type: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub max_position_embeddings: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    pub latent_channels: u32,
    pub encoder_channels: Vec<u32>,
    pub decoder_channels: Vec<u32>,
    pub time_compression_factor: u32,
}

#[derive(Debug)]
pub struct GGUFFile {
    pub config: GGUFConfig,
    pub tensors: Vec<TensorMetadata>,
    pub file: std::fs::File,
    pub mmap: memmap2::Mmap,
}

#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GGUFDType,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    UNKNOWN,
}

impl GGUFDType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GGUFDType::F32,
            1 => GGUFDType::F16,
            2 => GGUFDType::BF16,
            3 => GGUFDType::Q4_0,
            4 => GGUFDType::Q4_1,
            5 => GGUFDType::Q5_0,
            6 => GGUFDType::Q5_1,
            7 => GGUFDType::Q8_0,
            8 => GGUFDType::Q2_K,
            9 => GGUFDType::Q3_K,
            10 => GGUFDType::Q4_K,
            11 => GGUFDType::Q5_K,
            12 => GGUFDType::Q6_K,
            13 => GGUFDType::Q8_K,
            _ => GGUFDType::UNKNOWN,
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            GGUFDType::F32 => 4,
            GGUFDType::F16 => 2,
            GGUFDType::BF16 => 2,
            GGUFDType::Q4_0 => 2 + 16,
            GGUFDType::Q4_1 => 2 + 16 + 2,
            GGUFDType::Q5_0 => 2 + 4 + 16,
            GGUFDType::Q5_1 => 2 + 4 + 16 + 2,
            GGUFDType::Q8_0 => 2 + 32,
            GGUFDType::Q2_K => 256 / 16 + 256 / 4 + 2 + 2,
            GGUFDType::Q3_K => 256 / 8 + 256 / 4 + 12 + 2,
            GGUFDType::Q4_K => 2 + 2 + 12 + 256 / 2,
            GGUFDType::Q5_K => 2 + 2 + 12 + 256 / 2 + 256 / 8,
            GGUFDType::Q6_K => 256 / 2 + 256 / 4 + 256 / 16 + 2,
            GGUFDType::Q8_K => 4 + 4 + 12 + 256,
            GGUFDType::UNKNOWN => 4,
        }
    }
}

impl GGUFFile {
    pub fn load(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let metadata = std::fs::metadata(&file)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut reader = std::io::Cursor::new(&mmap);

        let magic = Self::read_string(&mut reader, 4)?;
        if magic != "GGUF" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: {}", magic),
            ));
        }

        let version: u32 = Self::read_val(&mut reader)?;
        let tensor_count: u64 = Self::read_val(&mut reader)?;
        let metadata_kv_count: u64 = Self::read_val(&mut reader)?;

        let mut general_config = GeneralConfig {
            architecture: String::new(),
            model_name: String::new(),
            model_version: String::new(),
            quantization_version: 0,
            alignment: 32,
            tensor_data_offset: 0,
        };

        let mut ltx_config = LTXVideoConfig {
            latent_channels: 16,
            in_channels: 16,
            out_channels: 16,
            time_embed_dim: 512,
            transform_layers: 28,
            num_attention_heads: 16,
            num_transformer_blocks: 28,
            frame_rate: 24,
            latent_height: 64,
            latent_width: 64,
            latent_frames: 9,
        };

        let mut text_config = TextEncoderConfig {
            model_type: String::new(),
            vocab_size: 0,
            hidden_size: 0,
            max_position_embeddings: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
        };

        let mut vae_config = VAEConfig {
            latent_channels: 16,
            encoder_channels: vec![128, 256, 512, 512],
            decoder_channels: vec![512, 512, 256, 128],
            time_compression_factor: 4,
        };

        let mut tensors = Vec::new();

        for _ in 0..metadata_kv_count {
            let key = Self::read_string_varint(&mut reader)?;
            let value_type: u32 = Self::read_val(&mut reader)?;

            match key.as_str() {
                "general.architecture" => {
                    general_config.architecture = Self::read_string_varint(&mut reader)?;
                }
                "general.name" => {
                    general_config.model_name = Self::read_string_varint(&mut reader)?;
                }
                "ltx_video.latent_channels" => {
                    ltx_config.latent_channels = Self::read_val(&mut reader)?;
                }
                "ltx_video.in_channels" => {
                    ltx_config.in_channels = Self::read_val(&mut reader)?;
                }
                "ltx_video.out_channels" => {
                    ltx_config.out_channels = Self::read_val(&mut reader)?;
                }
                "ltx_video.hidden_size" => {
                    ltx_config.time_embed_dim = Self::read_val(&mut reader)?;
                }
                "ltx_video.num_layers" => {
                    ltx_config.num_transformer_blocks = Self::read_val(&mut reader)?;
                }
                "ltx_video.num_attention_heads" => {
                    ltx_config.num_attention_heads = Self::read_val(&mut reader)?;
                }
                "ltx_video.frame_rate" => {
                    ltx_config.frame_rate = Self::read_val(&mut reader)?;
                }
                "ltx_video.latent_height" => {
                    ltx_config.latent_height = Self::read_val(&mut reader)?;
                }
                "ltx_video.latent_width" => {
                    ltx_config.latent_width = Self::read_val(&mut reader)?;
                }
                "ltx_video.latent_frames" => {
                    ltx_config.latent_frames = Self::read_val(&mut reader)?;
                }
                "text_encoder.vocab_size" => {
                    text_config.vocab_size = Self::read_val(&mut reader)?;
                }
                "text_encoder.hidden_size" => {
                    text_config.hidden_size = Self::read_val(&mut reader)?;
                }
                "text_encoder.num_layers" => {
                    text_config.num_hidden_layers = Self::read_val(&mut reader)?;
                }
                "vae.latent_channels" => {
                    vae_config.latent_channels = Self::read_val(&mut reader)?;
                }
                "general.quantization_version" => {
                    general_config.quantization_version = Self::read_val(&mut reader)?;
                }
                "general.alignment" => {
                    general_config.alignment = Self::read_val(&mut reader)?;
                }
                _ => match value_type {
                    0 => {
                        let _: u8 = Self::read_val(&mut reader)?;
                    }
                    1 => {
                        let _: u8 = Self::read_val(&mut reader)?;
                    }
                    2 => {
                        let _: u32 = Self::read_val(&mut reader)?;
                    }
                    3 => {
                        let _: u64 = Self::read_val(&mut reader)?;
                    }
                    4 => {
                        let _: i32 = Self::read_val(&mut reader)?;
                    }
                    5 => {
                        let _: i64 = Self::read_val(&mut reader)?;
                    }
                    6 => {
                        let _: f32 = Self::read_val(&mut reader)?;
                    }
                    7 => {
                        let _: f64 = Self::read_val(&mut reader)?;
                    }
                    8 => {
                        let _: bool = Self::read_val(&mut reader)?;
                    }
                    9 => {
                        let _: std::string::String = Self::read_string_varint(&mut reader)?;
                    }
                    10 => {
                        let _: Vec<u8> = Self::read_array_uint8(&mut reader)?;
                    }
                    11 => {
                        let _: Vec<i32> = Self::read_array_int32(&mut reader)?;
                    }
                    12 => {
                        let _: Vec<u64> = Self::read_array_uint64(&mut reader)?;
                    }
                    _ => {}
                },
            }
        }

        let tensor_data_offset = reader.position();

        for _ in 0..tensor_count {
            let name = Self::read_string_varint(&mut reader)?;
            let n_dims: u32 = Self::read_val(&mut reader)?;
            let dtype: u32 = Self::read_val(&mut reader)?;

            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(Self::read_val::<u64>(&mut reader)?);
            }

            let offset: u64 = Self::read_val(&mut reader)?;

            let gguf_dtype = GGUFDType::from_u32(dtype);
            let element_size = gguf_dtype.bytes_per_element();
            let num_elements: u64 = dims.iter().product();
            let size = num_elements * element_size as u64;

            tensors.push(TensorMetadata {
                name,
                dims,
                dtype: gguf_dtype,
                offset: tensor_data_offset + offset,
                size,
            });
        }

        Ok(Self {
            config: GGUFConfig {
                general: general_config,
                ltx_video: ltx_config,
                text_encoder: text_config,
                vae: vae_config,
            },
            tensors,
            file,
            mmap,
        })
    }

    fn read_string(
        reader: &mut std::io::Cursor<&Mmap>,
        len: usize,
    ) -> Result<String, std::io::Error> {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    fn read_string_varint(reader: &mut std::io::Cursor<&Mmap>) -> Result<String, std::io::Error> {
        let len = Self::read_val::<u64>(reader)? as usize;
        Self::read_string(reader, len)
    }

    fn read_val<T: bytemuck::FromBytes>(
        reader: &mut std::io::Cursor<&Mmap>,
    ) -> Result<T, std::io::Error> {
        let mut buf = [0u8; std::mem::size_of::<T>()];
        reader.read_exact(&mut buf)?;
        Ok(bytemuck::from_bytes(&buf))
    }

    fn read_array_uint8(reader: &mut std::io::Cursor<&Mmap>) -> Result<Vec<u8>, std::io::Error> {
        let len = Self::read_val::<u64>(reader)? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn read_array_int32(reader: &mut std::io::Cursor<&Mmap>) -> Result<Vec<i32>, std::io::Error> {
        let len = Self::read_val::<u64>(reader)? as usize;
        let mut buf = vec![0i32; len];
        for item in &mut buf {
            *item = Self::read_val(reader)?;
        }
        Ok(buf)
    }

    fn read_array_uint64(reader: &mut std::io::Cursor<&Mmap>) -> Result<Vec<u64>, std::io::Error> {
        let len = Self::read_val::<u64>(reader)? as usize;
        let mut buf = vec![0u64; len];
        for item in &mut buf {
            *item = Self::read_val(reader)?;
        }
        Ok(buf)
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn list_tensors(&self) -> &[TensorMetadata] {
        &self.tensors
    }

    pub fn load_tensor_data(&self, tensor: &TensorMetadata) -> Result<Vec<u8>, std::io::Error> {
        let offset = tensor.offset as usize;
        let size = tensor.size as usize;
        Ok(self.mmap[offset..offset + size].to_vec())
    }
}

mod bytemuck {
    pub trait FromBytes: Sized {
        fn from_bytes(bytes: &[u8]) -> Self;
    }

    macro_rules! impl_from_bytes {
        ($t:ty) => {
            impl FromBytes for $t {
                fn from_bytes(bytes: &[u8]) -> Self {
                    let mut arr = [0u8; std::mem::size_of::<$t>()];
                    arr.copy_from_slice(&bytes[..std::mem::size_of::<$t>()]);
                    <$t>::from_le_bytes(arr)
                }
            }
        };
    }

    impl_from_bytes!(u8);
    impl_from_bytes!(u32);
    impl_from_bytes!(u64);
    impl_from_bytes!(i32);
    impl_from_bytes!(i64);
    impl_from_bytes!(f32);
    impl_from_bytes!(f64);
    impl_from_bytes!(bool);
}

use memmap2::Mmap;
