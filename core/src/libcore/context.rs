pub struct Context {
    pub embeddings: super::Tensor,
    pub embeddings_neg: Option<super::Tensor>,
    pub seq_len: u32,
}

impl Context {
    pub fn new(embeddings: super::Tensor, seq_len: u32) -> Self {
        Self {
            embeddings,
            embeddings_neg: None,
            seq_len,
        }
    }

    pub fn with_negative(
        embeddings: super::Tensor,
        embeddings_neg: super::Tensor,
        seq_len: u32,
    ) -> Self {
        Self {
            embeddings,
            embeddings_neg: Some(embeddings_neg),
            seq_len,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SamplerType {
    Euler,
    EulerA,
    DDIM,
    DPMPlusPlus,
}

impl Default for SamplerType {
    fn default() -> Self {
        SamplerType::Euler
    }
}

impl std::fmt::Display for SamplerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerType::Euler => write!(f, "euler"),
            SamplerType::EulerA => write!(f, "euler_a"),
            SamplerType::DDIM => write!(f, "ddim"),
            SamplerType::DPMPlusPlus => write!(f, "dpm++"),
        }
    }
}

pub struct GenerateRequest<'a> {
    pub prompt: &'a str,
    pub negative_prompt: Option<&'a str>,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub fps: Option<usize>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub seed: Option<u64>,
    pub sampler: Option<SamplerType>,
    pub callback: Option<fn(usize, usize)>,
}

impl<'a> Clone for GenerateRequest<'a> {
    fn clone(&self) -> Self {
        Self {
            prompt: self.prompt,
            negative_prompt: self.negative_prompt,
            frames: self.frames,
            width: self.width,
            height: self.height,
            fps: self.fps,
            steps: self.steps,
            cfg_scale: self.cfg_scale,
            seed: self.seed,
            sampler: self.sampler.clone(),
            callback: self.callback,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub frames: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub generation_time_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub total_steps: usize,
    pub current_step: usize,
    pub time_per_step_ms: u64,
    pub total_time_ms: u64,
    pub vram_allocated_mb: u64,
    pub vram_reserved_mb: u64,
}
