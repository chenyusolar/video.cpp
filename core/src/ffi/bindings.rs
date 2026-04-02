use crate::config::Config;
use crate::libcore::context::Context;
use crate::libcore::tensor::{DType, Tensor, TensorData, TensorShape};
use crate::libcore::traits::Backend;
use crate::libcore::traits::{Error, Model, Result, TextEncoder, VAE};
use crate::model::ModelLoader;
use crate::pipeline::{GenerateRequest, VideoOutput, VideoPipeline};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub type model_handle = u64;
pub type tensor_handle = u64;

static mut MODEL_REGISTRY: Option<HashMap<model_handle, Arc<Mutex<VideoPipeline>>>> = None;
static mut NEXT_HANDLE: model_handle = 1;

fn get_registry() -> &'static mut HashMap<model_handle, Arc<Mutex<VideoPipeline>>> {
    unsafe {
        if MODEL_REGISTRY.is_none() {
            MODEL_REGISTRY = Some(HashMap::new());
        }
        MODEL_REGISTRY.as_mut().unwrap()
    }
}

fn next_handle() -> model_handle {
    unsafe {
        let h = NEXT_HANDLE;
        NEXT_HANDLE += 1;
        h
    }
}

#[repr(C)]
pub struct generate_request {
    pub prompt: *const std::os::raw::c_char,
    pub negative_prompt: *const std::os::raw::c_char,
    pub frames: i32,
    pub width: i32,
    pub height: i32,
    pub fps: i32,
    pub steps: i32,
    pub sampler: *const std::os::raw::c_char,
    pub cfg_scale: f32,
    pub seed: i64,
    pub device_id: i32,
}

#[repr(C)]
pub struct video_output {
    pub data: *mut u8,
    pub size: usize,
    pub width: i32,
    pub height: i32,
    pub fps: i32,
    pub generation_time_ms: i64,
}

#[repr(i32)]
pub enum video_backend {
    VIDEO_BACKEND_CPU = 0,
    VIDEO_BACKEND_CUDA = 1,
    VIDEO_BACKEND_VULKAN = 2,
}

#[repr(i32)]
#[derive(Debug, PartialEq)]
pub enum video_error {
    VIDEO_OK = 0,
    VIDEO_ERROR_LOAD_FAILED = 1,
    VIDEO_ERROR_INVALID_PARAM = 2,
    VIDEO_ERROR_GENERATION_FAILED = 3,
    VIDEO_ERROR_OUT_OF_MEMORY = 4,
    VIDEO_ERROR_BACKEND_ERROR = 5,
    VIDEO_ERROR_UNSUPPORTED = 6,
}

#[no_mangle]
pub unsafe extern "C" fn video_load(
    model_path: *const std::os::raw::c_char,
    out_handle: *mut model_handle,
) -> video_error {
    if model_path.is_null() || out_handle.is_null() {
        return video_error::VIDEO_ERROR_INVALID_PARAM;
    }

    let path = std::ffi::CStr::from_ptr(model_path)
        .to_string_lossy()
        .into_owned();

    let config = Config::from_env();

    let backend: Arc<dyn Backend> = match config.backend.backend_type {
        crate::config::BackendType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Arc::new(
                    crate::backend::cuda::CudaBackend::new(config.backend.device_id)
                        .unwrap_or_else(|_| crate::backend::cpu::CpuBackend::new()),
                ) as Arc<dyn Backend>
            }
            #[cfg(not(feature = "cuda"))]
            {
                Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>
            }
        }
        crate::config::BackendType::Vulkan => {
            Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>
        }
        _ => Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>,
    };

    let pipeline = match VideoPipeline::new(&path, backend) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create pipeline: {}", e);
            return video_error::VIDEO_ERROR_LOAD_FAILED;
        }
    };

    let handle = next_handle();
    let registry = get_registry();
    registry.insert(handle, Arc::new(Mutex::new(pipeline)));

    *out_handle = handle;
    tracing::info!("Loaded model with handle: {}", handle);
    video_error::VIDEO_OK
}

#[no_mangle]
pub unsafe extern "C" fn video_free(handle: model_handle) -> video_error {
    let registry = get_registry();
    registry.remove(&handle);
    tracing::info!("Freed model handle: {}", handle);
    video_error::VIDEO_OK
}

#[no_mangle]
pub unsafe extern "C" fn video_generate(
    handle: model_handle,
    req: generate_request,
    out: *mut video_output,
) -> video_error {
    if out.is_null() {
        return video_error::VIDEO_ERROR_INVALID_PARAM;
    }

    let registry = get_registry();
    let pipeline_arc = match registry.get(&handle) {
        Some(h) => h,
        None => {
            tracing::error!("Invalid model handle: {}", handle);
            return video_error::VIDEO_ERROR_LOAD_FAILED;
        }
    };

    let mut pipeline = match pipeline_arc.lock() {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to lock pipeline: {}", e);
            return video_error::VIDEO_ERROR_GENERATION_FAILED;
        }
    };

    let prompt = if req.prompt.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(req.prompt)
            .to_string_lossy()
            .into_owned()
    };

    let negative_prompt = if req.negative_prompt.is_null() {
        None
    } else {
        Some(
            std::ffi::CStr::from_ptr(req.negative_prompt)
                .to_string_lossy()
                .into_owned(),
        )
    };

    let sampler = if req.sampler.is_null() {
        "euler".to_string()
    } else {
        std::ffi::CStr::from_ptr(req.sampler)
            .to_string_lossy()
            .into_owned()
    };

    let pipeline_req = GenerateRequest {
        prompt: prompt,
        negative_prompt: negative_prompt,
        frames: req.frames as usize,
        width: req.width as usize,
        height: req.height as usize,
        fps: Some(req.fps as usize),
        steps: Some(req.steps as u32),
        cfg_scale: Some(req.cfg_scale),
        seed: if req.seed >= 0 {
            Some(req.seed as u64)
        } else {
            None
        },
        callback: None,
    };

    tracing::info!(
        "Starting generation: {}x{} {} frames, {} steps",
        req.width,
        req.height,
        req.frames,
        req.steps
    );

    let result = pipeline.generate(&pipeline_req);

    match result {
        Ok(video) => {
            let data = video.frames.into_boxed_slice();
            let size = data.len();
            let ptr = Box::into_raw(data) as *mut u8;

            *out = video_output {
                data: ptr,
                size,
                width: video.width as i32,
                height: video.height as i32,
                fps: video.fps as i32,
                generation_time_ms: video.generation_time_ms as i64,
            };
            tracing::info!("Generation complete in {}ms", video.generation_time_ms);
            video_error::VIDEO_OK
        }
        Err(e) => {
            tracing::error!("Generation failed: {}", e);
            video_error::VIDEO_ERROR_GENERATION_FAILED
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn video_set_backend(backend: video_backend) -> video_error {
    match backend {
        video_backend::VIDEO_BACKEND_CPU => {
            std::env::set_var("VIDEO_BACKEND", "cpu");
        }
        video_backend::VIDEO_BACKEND_CUDA => {
            std::env::set_var("VIDEO_BACKEND", "cuda");
        }
        video_backend::VIDEO_BACKEND_VULKAN => {
            std::env::set_var("VIDEO_BACKEND", "vulkan");
        }
    }
    video_error::VIDEO_OK
}

#[no_mangle]
pub extern "C" fn video_get_version() -> *const std::os::raw::c_char {
    b"video.cpp 0.1.0\0".as_ptr() as *const std::os::raw::c_char
}

#[no_mangle]
pub unsafe extern "C" fn video_get_memory_info(
    allocated_bytes: *mut i64,
    reserved_bytes: *mut i64,
) -> video_error {
    if allocated_bytes.is_null() || reserved_bytes.is_null() {
        return video_error::VIDEO_ERROR_INVALID_PARAM;
    }

    *allocated_bytes = 0;
    *reserved_bytes = 0;

    video_error::VIDEO_OK
}

#[no_mangle]
pub unsafe extern "C" fn video_generate_image_to_video(
    handle: model_handle,
    init_image: *const u8,
    image_size: usize,
    prompt: *const std::os::raw::c_char,
    width: i32,
    height: i32,
    frames: i32,
    strength: f32,
    steps: i32,
    cfg_scale: f32,
    seed: i64,
    out: *mut video_output,
) -> video_error {
    if out.is_null() {
        return video_error::VIDEO_ERROR_INVALID_PARAM;
    }

    let registry = get_registry();
    let pipeline_arc = match registry.get(&handle) {
        Some(h) => h,
        None => return video_error::VIDEO_ERROR_LOAD_FAILED,
    };

    let mut pipeline = match pipeline_arc.lock() {
        Ok(p) => p,
        Err(_) => return video_error::VIDEO_ERROR_GENERATION_FAILED,
    };

    let prompt_str = if prompt.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(prompt)
            .to_string_lossy()
            .into_owned()
    };

    let image_data = if !init_image.is_null() {
        Some(std::slice::from_raw_parts(init_image, image_size))
    } else {
        None
    };

    tracing::info!("Image-to-Video: {}x{}, {} frames", width, height, frames);

    let result = pipeline.image_to_video(
        image_data.unwrap_or(&[]),
        &prompt_str,
        width as u32,
        height as u32,
        frames as u32,
        strength,
        steps as u32,
        cfg_scale,
        if seed >= 0 { Some(seed as u64) } else { None },
    );

    match result {
        Ok(video) => {
            let data = video.frames.into_boxed_slice();
            let size = data.len();
            let ptr = Box::into_raw(data) as *mut u8;

            *out = video_output {
                data: ptr,
                size,
                width: video.width as i32,
                height: video.height as i32,
                fps: video.fps as i32,
                generation_time_ms: video.generation_time_ms as i64,
            };
            video_error::VIDEO_OK
        }
        Err(_) => video_error::VIDEO_ERROR_GENERATION_FAILED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn video_generate_video_to_video(
    handle: model_handle,
    init_video: *const u8,
    video_size: usize,
    prompt: *const std::os::raw::c_char,
    width: i32,
    height: i32,
    frames: i32,
    strength: f32,
    steps: i32,
    cfg_scale: f32,
    seed: i64,
    out: *mut video_output,
) -> video_error {
    if out.is_null() {
        return video_error::VIDEO_ERROR_INVALID_PARAM;
    }

    let registry = get_registry();
    let pipeline_arc = match registry.get(&handle) {
        Some(h) => h,
        None => return video_error::VIDEO_ERROR_LOAD_FAILED,
    };

    let mut pipeline = match pipeline_arc.lock() {
        Ok(p) => p,
        Err(_) => return video_error::VIDEO_ERROR_GENERATION_FAILED,
    };

    let prompt_str = if prompt.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(prompt)
            .to_string_lossy()
            .into_owned()
    };

    let video_data = if !init_video.is_null() {
        Some(std::slice::from_raw_parts(init_video, video_size))
    } else {
        None
    };

    tracing::info!("Video-to-Video: {}x{}, {} frames", width, height, frames);

    let result = pipeline.video_to_video(
        video_data.unwrap_or(&[]),
        &prompt_str,
        width as u32,
        height as u32,
        frames as u32,
        strength,
        steps as u32,
        cfg_scale,
        if seed >= 0 { Some(seed as u64) } else { None },
    );

    match result {
        Ok(video) => {
            let data = video.frames.into_boxed_slice();
            let size = data.len();
            let ptr = Box::into_raw(data) as *mut u8;

            *out = video_output {
                data: ptr,
                size,
                width: video.width as i32,
                height: video.height as i32,
                fps: video.fps as i32,
                generation_time_ms: video.generation_time_ms as i64,
            };
            video_error::VIDEO_OK
        }
        Err(_) => video_error::VIDEO_ERROR_GENERATION_FAILED,
    }
}
