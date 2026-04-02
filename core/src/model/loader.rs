use std::sync::Arc;

use crate::libcore::tensor::{Tensor, TensorShape};
use crate::libcore::traits::{Error, Result};

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFVIDLoader {
    mmap: Arc<memmap2::Mmap>,
    metadata: GGUFMetadata,
    tensor_infos: Vec<TensorInfo>,
    kv_data: std::collections::HashMap<String, String>,
}

impl GGUFVIDLoader {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io(e))?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).map_err(|e| Error::Io(e))? });

        let mut loader = Self {
            mmap,
            metadata: GGUFMetadata {
                magic: 0,
                version: 0,
                tensor_count: 0,
                kv_count: 0,
            },
            tensor_infos: Vec::new(),
            kv_data: std::collections::HashMap::new(),
        };

        loader.read_header()?;
        loader.read_kv_metadata()?;
        loader.read_tensor_infos()?;

        Ok(loader)
    }

    fn read_header(&mut self) -> Result<()> {
        if self.mmap.len() < 24 {
            return Err(Error::Model("File too small for GGUF header".into()));
        }

        let magic = u32::from_le_bytes([self.mmap[0], self.mmap[1], self.mmap[2], self.mmap[3]]);
        if magic != GGUF_MAGIC {
            return Err(Error::Model(format!(
                "Invalid GGUF magic: {:x}, expected: {:x}",
                magic, GGUF_MAGIC
            )));
        }

        let version = u32::from_le_bytes([self.mmap[4], self.mmap[5], self.mmap[6], self.mmap[7]]);
        if version != GGUF_VERSION {
            tracing::warn!(
                "GGUF version {} not fully tested, expected {}",
                version,
                GGUF_VERSION
            );
        }

        let tensor_count = u64::from_le_bytes([
            self.mmap[8],
            self.mmap[9],
            self.mmap[10],
            self.mmap[11],
            self.mmap[12],
            self.mmap[13],
            self.mmap[14],
            self.mmap[15],
        ]);

        let kv_count = u64::from_le_bytes([
            self.mmap[16],
            self.mmap[17],
            self.mmap[18],
            self.mmap[19],
            self.mmap[20],
            self.mmap[21],
            self.mmap[22],
            self.mmap[23],
        ]);

        tracing::info!(
            "GGUF Header: magic=0x{:x}, version={}, tensors={}, kvs={}",
            magic,
            version,
            tensor_count,
            kv_count
        );

        self.metadata.magic = magic;
        self.metadata.version = version;
        self.metadata.tensor_count = tensor_count;
        self.metadata.kv_count = kv_count;

        Ok(())
    }

    fn read_kv_metadata(&mut self) -> Result<()> {
        let mut offset = 24;

        for _ in 0..self.metadata.kv_count {
            if offset >= self.mmap.len() {
                break;
            }

            let (key, new_offset) = self.read_string(offset)?;
            offset = new_offset;

            if offset + 4 > self.mmap.len() {
                break;
            }
            let value_type = u32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]);
            offset += 4;

            let (value, new_offset) = self.read_value(offset, value_type)?;
            offset = new_offset;

            self.kv_data.insert(key.clone(), value.clone());

            if key.starts_with("general.") || key.starts_with("ltx-video.") {
                tracing::debug!("KV: {} = {:?}", key, value);
            }
        }

        Ok(())
    }

    fn read_string(&self, offset: usize) -> Result<(String, usize)> {
        if offset + 8 > self.mmap.len() {
            return Err(Error::Model("Unexpected EOF reading string length".into()));
        }

        let len = u64::from_le_bytes([
            self.mmap[offset],
            self.mmap[offset + 1],
            self.mmap[offset + 2],
            self.mmap[offset + 3],
            self.mmap[offset + 4],
            self.mmap[offset + 5],
            self.mmap[offset + 6],
            self.mmap[offset + 7],
        ]) as usize;

        let mut new_offset = offset + 8;
        if new_offset + len > self.mmap.len() {
            return Err(Error::Model("String data out of bounds".into()));
        }

        let s = String::from_utf8_lossy(&self.mmap[new_offset..new_offset + len]).to_string();
        new_offset += len;

        Ok((s, new_offset))
    }

    fn read_value(&self, offset: usize, value_type: u32) -> Result<(String, usize)> {
        match value_type {
            0 => {
                let val = self.mmap[offset];
                Ok((format!("{}", val), offset + 1))
            }
            1 => {
                let val = self.mmap[offset] as i8;
                Ok((format!("{}", val), offset + 1))
            }
            2 => {
                let val = u32::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                ]);
                Ok((format!("{}", val), offset + 4))
            }
            3 => {
                let val = i32::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                ]);
                Ok((format!("{}", val), offset + 4))
            }
            4 => {
                let val = f32::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                ]);
                Ok((format!("{}", val), offset + 4))
            }
            5 => {
                let val = self.mmap[offset] != 0;
                Ok((format!("{}", val), offset + 1))
            }
            6 => {
                let val = u64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]);
                Ok((format!("{}", val), offset + 8))
            }
            7 => {
                let val = i64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]);
                Ok((format!("{}", val), offset + 8))
            }
            8 => self.read_string(offset),
            9 => {
                let len = u64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]) as usize;
                Ok((format!("Vec<u8>[{}]", len), offset + 8 + len))
            }
            10 => {
                let len = u64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]) as usize;
                Ok((format!("Vec<i32>[{}]", len), offset + 8 + len * 4))
            }
            11 => {
                let len = u64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]) as usize;
                Ok((format!("Vec<u64>[{}]", len), offset + 8 + len * 8))
            }
            _ => Ok((format!("<type={}>", value_type), offset)),
        }
    }

    fn read_tensor_infos(&mut self) -> Result<()> {
        let mut offset = 24 + self.compute_kv_metadata_size();

        offset = Self::align_offset(offset, GGUF_DEFAULT_ALIGNMENT as usize);

        tracing::info!(
            "Reading {} tensors starting at offset {}",
            self.metadata.tensor_count,
            offset
        );

        for i in 0..self.metadata.tensor_count {
            if offset >= self.mmap.len() {
                break;
            }

            let (name, new_offset) = self.read_string(offset)?;
            offset = new_offset;

            if offset + 4 > self.mmap.len() {
                break;
            }
            let n_dims = u32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]);
            offset += 4;

            let mut shape = Vec::new();
            let mut total_elems: u64 = 1;
            for _ in 0..n_dims {
                if offset + 4 > self.mmap.len() {
                    break;
                }
                let dim = u32::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                ]);
                total_elems = total_elems.saturating_mul(dim as u64);
                shape.push(dim as u64);
                offset += 4;
            }

            if offset + 4 > self.mmap.len() {
                break;
            }
            let dtype = u32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]);
            offset += 4;

            if offset + 8 > self.mmap.len() {
                break;
            }
            let tensor_offset = u64::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
                self.mmap[offset + 4],
                self.mmap[offset + 5],
                self.mmap[offset + 6],
                self.mmap[offset + 7],
            ]);
            offset += 8;

            self.tensor_infos.push(TensorInfo {
                name,
                n_dims,
                shape,
                dtype,
                offset: tensor_offset,
            });

            if i % 100 == 0 {
                tracing::debug!(
                    "Read tensor {}: name={}, n_dims={}, dtype={}, offset={}",
                    i,
                    self.tensor_infos.last().unwrap().name,
                    n_dims,
                    dtype,
                    tensor_offset
                );
            }
        }

        tracing::info!("Loaded {} tensor infos", self.tensor_infos.len());
        Ok(())
    }

    fn compute_kv_metadata_size(&self) -> usize {
        let mut offset = 24;
        for _ in 0..self.metadata.kv_count.min(100) {
            if offset >= self.mmap.len().min(offset + 1000) {
                break;
            }
            if offset + 8 > self.mmap.len() {
                break;
            }
            let len = u64::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
                self.mmap[offset + 4],
                self.mmap[offset + 5],
                self.mmap[offset + 6],
                self.mmap[offset + 7],
            ]) as usize;
            offset += 8 + len + 4;
        }
        offset
    }

    fn align_offset(offset: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return offset;
        }
        let rem = offset % alignment;
        if rem == 0 {
            offset
        } else {
            offset + (alignment - rem)
        }
    }

    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    pub fn get_kv(&self, key: &str) -> Option<&str> {
        self.kv_data.get(key).map(|s| s.as_str())
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| Error::NotFound(format!("Tensor not found: {}", name)))?;

        self.load_tensor_by_info(info)
    }

    pub fn load_tensor_by_info(&self, info: &TensorInfo) -> Result<Tensor> {
        let offset = info.offset as usize;

        let elem_count: u64 = info.shape.iter().product();
        if elem_count == 0 {
            return Err(Error::Model("Tensor has zero elements".into()));
        }
        if elem_count > 1_000_000_000 {
            return Err(Error::Model(format!(
                "Tensor {} has too many elements: {}",
                info.name, elem_count
            )));
        }

        let dtype_size = match info.dtype {
            0 => 4,
            1 => 2,
            2 => 1,
            _ => return Err(Error::Model(format!("Unsupported dtype: {}", info.dtype))),
        };

        let size = (elem_count as usize).saturating_mul(dtype_size);

        if offset >= self.mmap.len() {
            return Err(Error::Model(format!(
                "Tensor {} offset {} >= file size {}",
                info.name,
                offset,
                self.mmap.len()
            )));
        }
        if offset + size > self.mmap.len() {
            return Err(Error::Model(format!(
                "Tensor {} data out of bounds: offset={}, size={}, file_len={}",
                info.name,
                offset,
                size,
                self.mmap.len()
            )));
        }

        let data = &self.mmap[offset..offset + size];

        let shape = TensorShape::new(info.shape.iter().map(|&x| x as u32).collect());

        let tensor_data = match info.dtype {
            0 => crate::libcore::tensor::TensorData::F32(
                data.chunks(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect(),
            ),
            1 => crate::libcore::tensor::TensorData::F16(
                data.chunks(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect(),
            ),
            _ => return Err(Error::Model(format!("Unsupported dtype: {}", info.dtype))),
        };

        Ok(Tensor::from_data(shape, tensor_data))
    }

    pub fn mmap_tensor(&self, name: &str) -> Result<std::borrow::Cow<[u8]>> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| Error::NotFound(format!("Tensor not found: {}", name)))?;

        let offset = info.offset as usize;

        let elem_count: u64 = info.shape.iter().product();
        if elem_count == 0 || elem_count > 1_000_000_000 {
            return Err(Error::Model("Tensor has invalid size".into()));
        }

        let dtype_size = match info.dtype {
            0 => 4,
            1 => 2,
            2 => 1,
            _ => return Err(Error::Model(format!("Unsupported dtype: {}", info.dtype))),
        };

        let size = (elem_count as usize).saturating_mul(dtype_size);

        if offset + size > self.mmap.len() {
            return Err(Error::Model("Tensor data out of bounds".into()));
        }

        Ok(std::borrow::Cow::Borrowed(
            &self.mmap[offset..offset + size],
        ))
    }

    pub fn list_tensors(&self) -> Vec<&TensorInfo> {
        self.tensor_infos.iter().collect()
    }
}
