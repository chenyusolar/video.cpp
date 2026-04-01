use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::ffi::bindings::{
    generate_request, model_handle, tensor_handle, video_backend, video_backend_to_c,
    video_free as c_video_free, video_generate as c_video_generate,
    video_get_version as c_video_get_version, video_load as c_video_load, video_output,
    video_set_backend as c_video_set_backend,
};
use crate::ffi::error::Error as FfiError;

pub struct VideoEngine {
    handle: model_handle,
    config: Config,
}

impl VideoEngine {
    pub fn load(path: &str) -> Result<Self, FfiError> {
        let c_path = CString::new(path)
            .map_err(|_| FfiError::InvalidParameter("Invalid model path".into()))?;

        let mut handle: model_handle = 0;

        let err = unsafe { c_video_load(c_path.as_ptr(), &mut handle) };
        if err != 0 {
            return Err(FfiError::LoadError(format!(
                "Failed to load model: error {}",
                err
            )));
        }

        let config = Config::from_env();

        Ok(Self { handle, config })
    }

    pub fn generate(&self, request: GenerateRequest) -> Result<VideoOutput, FfiError> {
        let c_request = request.to_c();
        let mut c_output = video_output {
            data: std::ptr::null_mut(),
            size: 0,
            width: 0,
            height: 0,
            fps: 0,
        };

        let err = unsafe { c_video_generate(self.handle, c_request, &mut c_output) };
        if err != 0 {
            return Err(FfiError::GenerateError(format!(
                "Generation failed: error {}",
                err
            )));
        }

        let data = if c_output.data.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(c_output.data, c_output.size).to_vec() }
        };

        Ok(VideoOutput {
            data,
            width: c_output.width as usize,
            height: c_output.height as usize,
            fps: c_output.fps as usize,
        })
    }

    pub fn set_backend(backend: BackendType) -> Result<(), FfiError> {
        let c_backend = video_backend_to_c(backend);
        let err = unsafe { c_video_set_backend(c_backend) };
        if err != 0 {
            return Err(FfiError::BackendError(format!(
                "Failed to set backend: error {}",
                err
            )));
        }
        Ok(())
    }

    pub fn version() -> String {
        unsafe {
            let c_version = c_video_get_version();
            CStr::from_ptr(c_version).to_string_lossy().into_owned()
        }
    }

    pub fn close(&mut self) {
        if self.handle != 0 {
            unsafe { c_video_free(self.handle) };
            self.handle = 0;
        }
    }
}

impl Drop for VideoEngine {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BackendType {
    CPU,
    CUDA,
    Vulkan,
}

fn video_backend_to_c(backend: BackendType) -> video_backend {
    match backend {
        BackendType::CPU => video_backend::VIDEO_BACKEND_CPU,
        BackendType::CUDA => video_backend::VIDEO_BACKEND_CUDA,
        BackendType::Vulkan => video_backend::VIDEO_BACKEND_VULKAN,
    }
}

pub struct GenerateRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub steps: usize,
    pub sampler: String,
    pub cfg_scale: f32,
    pub seed: Option<i64>,
    pub backend: Option<String>,
}

impl GenerateRequest {
    fn to_c(&self) -> generate_request {
        let prompt = CString::new(self.prompt.clone()).unwrap_or_default();
        let negative_prompt = self
            .negative_prompt
            .as_ref()
            .map(|s| CString::new(s.clone()).unwrap_or_default())
            .unwrap_or_default();
        let sampler = CString::new(self.sampler.clone()).unwrap_or_default();

        generate_request {
            prompt: prompt.as_ptr(),
            negative_prompt: negative_prompt.as_ptr(),
            frames: self.frames as i32,
            width: self.width as i32,
            height: self.height as i32,
            fps: self.fps as i32,
            steps: self.steps as i32,
            sampler: sampler.as_ptr(),
            cfg_scale: self.cfg_scale,
            seed: self.seed.unwrap_or(-1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoOutput {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub fps: usize,
}
