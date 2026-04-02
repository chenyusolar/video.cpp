use crate::libcore::tensor::{DType, Tensor, TensorShape};
use crate::libcore::traits::{Error, Result};

pub struct GGUFVIDLoader {
    file: std::fs::File,
    mmap: memmap2::Mmap,
    metadata: GGUFMetadata,
    tensor_infos: Vec<TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub magic: String,
    pub version: u32,
    pub tensor_count: u64,
    pub model_type: String,
    pub latent_shape: Vec<u32>,
    pub latent_channels: u32,
    pub fps: u32,
    pub has_audio: bool,
    pub vae_encoder: bool,
    pub vae_decoder: bool,
    pub text_encoder: bool,
    pub audio_encoder: bool,
    pub quantization_type: String,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum GGUV_dtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4,
    Q8_0 = 5,
}

impl GGUFVIDLoader {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| Error::Io(e))?;

        let metadata = file.metadata().map_err(|e| Error::Io(e))?;

        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| Error::Io(e))? };

        let mut loader = Self {
            file,
            mmap,
            metadata: GGUFMetadata::default(),
            tensor_infos: Vec::new(),
        };

        loader.read_header()?;
        loader.read_metadata()?;
        loader.read_tensor_infos()?;

        Ok(loader)
    }

    fn read_header(&mut self) -> Result<()> {
        if self.mmap.len() < 12 {
            return Err(Error::Model("File too small for header".into()));
        }

        let magic = String::from_utf8_lossy(&self.mmap[0..4]).to_string();
        if magic != "GGUV" {
            return Err(Error::Model(format!("Invalid magic: {}", magic)));
        }

        let version = u32::from_le_bytes([self.mmap[4], self.mmap[5], self.mmap[6], self.mmap[7]]);

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

        self.metadata.magic = magic;
        self.metadata.version = version;
        self.metadata.tensor_count = tensor_count;

        Ok(())
    }

    fn read_metadata(&mut self) -> Result<()> {
        let offset = 16;

        let mut pos = offset;

        let model_type_len = self.mmap[pos] as usize;
        pos += 1;
        if pos + model_type_len > self.mmap.len() {
            return Err(Error::Model("Invalid metadata: model_type".into()));
        }
        self.metadata.model_type =
            String::from_utf8_lossy(&self.mmap[pos..pos + model_type_len]).to_string();
        pos += model_type_len;

        let latent_shape_len = self.mmap[pos] as usize;
        pos += 1;
        self.metadata.latent_shape = Vec::new();
        for _ in 0..latent_shape_len {
            let val = u32::from_le_bytes([
                self.mmap[pos],
                self.mmap[pos + 1],
                self.mmap[pos + 2],
                self.mmap[pos + 3],
            ]);
            self.metadata.latent_shape.push(val);
            pos += 4;
        }

        self.metadata.latent_channels = u32::from_le_bytes([
            self.mmap[pos],
            self.mmap[pos + 1],
            self.mmap[pos + 2],
            self.mmap[pos + 3],
        ]);
        pos += 4;

        self.metadata.fps = u32::from_le_bytes([
            self.mmap[pos],
            self.mmap[pos + 1],
            self.mmap[pos + 2],
            self.mmap[pos + 3],
        ]);
        pos += 4;

        self.metadata.has_audio = self.mmap[pos] != 0;
        pos += 1;
        self.metadata.vae_encoder = self.mmap[pos] != 0;
        pos += 1;
        self.metadata.vae_decoder = self.mmap[pos] != 0;
        pos += 1;
        self.metadata.text_encoder = self.mmap[pos] != 0;
        pos += 1;
        self.metadata.audio_encoder = self.mmap[pos] != 0;
        pos += 1;

        let quant_len = self.mmap[pos] as usize;
        pos += 1;
        if pos + quant_len > self.mmap.len() {
            return Err(Error::Model("Invalid metadata: quantization".into()));
        }
        self.metadata.quantization_type =
            String::from_utf8_lossy(&self.mmap[pos..pos + quant_len]).to_string();

        Ok(())
    }

    fn read_tensor_infos(&mut self) -> Result<()> {
        let mut offset = 16 + 256;

        for _ in 0..self.metadata.tensor_count {
            if offset >= self.mmap.len() {
                break;
            }

            let name_len = self.mmap[offset] as usize;
            offset += 1;
            if offset + name_len > self.mmap.len() {
                break;
            }
            let name = String::from_utf8_lossy(&self.mmap[offset..offset + name_len]).to_string();
            offset += name_len;

            let n_dims = u32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]);
            offset += 4;

            let mut shape = Vec::new();
            for _ in 0..n_dims {
                let dim = u64::from_le_bytes([
                    self.mmap[offset],
                    self.mmap[offset + 1],
                    self.mmap[offset + 2],
                    self.mmap[offset + 3],
                    self.mmap[offset + 4],
                    self.mmap[offset + 5],
                    self.mmap[offset + 6],
                    self.mmap[offset + 7],
                ]);
                shape.push(dim);
                offset += 8;
            }

            let dtype = u32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]);
            offset += 4;

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
        }

        Ok(())
    }

    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| Error::NotFound(format!("Tensor not found: {}", name)))?;

        let offset = info.offset as usize;
        let size = info.shape.iter().product::<u64>() as usize * 4;

        if offset + size > self.mmap.len() {
            return Err(Error::Model(format!(
                "Tensor {} data out of bounds: offset={}, size={}, len={}",
                name,
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
        let size = info.shape.iter().product::<u64>() as usize * 4;

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

impl Default for GGUFMetadata {
    fn default() -> Self {
        Self {
            magic: String::new(),
            version: 1,
            tensor_count: 0,
            model_type: "ltx-video".to_string(),
            latent_shape: vec![1, 16, 64, 64],
            latent_channels: 16,
            fps: 24,
            has_audio: false,
            vae_encoder: true,
            vae_decoder: true,
            text_encoder: true,
            audio_encoder: false,
            quantization_type: "none".to_string(),
        }
    }
}
