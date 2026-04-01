use std::io::Write;
use std::path::Path;

pub struct VideoEncoder {
    width: u32,
    height: u32,
    fps: u32,
    codec: VideoCodec,
}

#[derive(Debug, Clone, Copy)]
pub enum VideoCodec {
    H264,
    H265,
    VP9,
    AV1,
    RawRGB,
    RawRGBA,
}

impl VideoEncoder {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self {
            width,
            height,
            fps,
            codec: VideoCodec::H264,
        }
    }

    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    pub fn encode_frames(&self, frames: &[u8], num_frames: usize) -> Result<Vec<u8>, EncodeError> {
        let frame_size = (self.width * self.height * 3) as usize;
        let expected_size = frame_size * num_frames;

        if frames.len() < expected_size {
            return Err(EncodeError::InvalidInput(format!(
                "Expected {} bytes for {} frames, got {}",
                expected_size,
                num_frames,
                frames.len()
            )));
        }

        match self.codec {
            VideoCodec::RawRGB => self.encode_raw_rgb(frames, num_frames),
            VideoCodec::RawRGBA => self.encode_raw_rgba(frames, num_frames),
            VideoCodec::H264 => self.encode_with_ffmpeg(frames, "libx264", num_frames),
            VideoCodec::H265 => self.encode_with_ffmpeg(frames, "libx265", num_frames),
            VideoCodec::VP9 => self.encode_with_ffmpeg(frames, "libvpx-vp9", num_frames),
            VideoCodec::AV1 => self.encode_with_ffmpeg(frames, "libaom-av1", num_frames),
        }
    }

    fn encode_raw_rgb(&self, frames: &[u8], num_frames: usize) -> Result<Vec<u8>, EncodeError> {
        let header = self.create_bmp_header(num_frames);
        let mut output = Vec::with_capacity(header.len() + frames.len() + num_frames * 12);
        output.extend_from_slice(&header);
        output.extend_from_slice(frames);
        Ok(output)
    }

    fn encode_raw_rgba(&self, frames: &[u8], num_frames: usize) -> Result<Vec<u8>, EncodeError> {
        let mut output = Vec::with_capacity(frames.len() * 4 / 3);

        for chunk in frames.chunks((self.width * self.height * 3) as usize) {
            for i in 0..chunk.len() / 3 {
                output.push(chunk[i * 3]);
                output.push(chunk[i * 3 + 1]);
                output.push(chunk[i * 3 + 2]);
                output.push(255);
            }
        }

        Ok(output)
    }

    fn encode_with_ffmpeg(
        &self,
        frames: &[u8],
        codec: &str,
        num_frames: usize,
    ) -> Result<Vec<u8>, EncodeError> {
        use std::io::Read;
        use std::process::{Command, Stdio};

        let temp_dir = std::env::temp_dir();
        let input_file = temp_dir.join("video_input_raw.rgb");
        let output_file = temp_dir.join("video_output.mp4");

        std::fs::write(&input_file, frames)
            .map_err(|e| EncodeError::IoError(format!("Failed to write temp input: {}", e)))?;

        let width = self.width.to_string();
        let height = self.height.to_string();
        let fps = self.fps.to_string();

        let ffmpeg_result = Command::new("ffmpeg")
            .args(&[
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &format!("{}x{}", width, height),
                "-framerate",
                &fps,
                "-i",
                input_file.to_str().unwrap(),
                "-c:v",
                codec,
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-y",
                output_file.to_str().unwrap(),
            ])
            .output();

        let _ = std::fs::remove_file(&input_file);

        match ffmpeg_result {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    tracing::warn!("FFmpeg failed: {}, falling back to raw output", stderr);
                    return self.encode_raw_rgb(frames, num_frames);
                }

                match std::fs::read(&output_file) {
                    Ok(data) => {
                        let _ = std::fs::remove_file(&output_file);
                        Ok(data)
                    }
                    Err(e) => {
                        let _ = std::fs::remove_file(&output_file);
                        Err(EncodeError::IoError(format!(
                            "Failed to read ffmpeg output: {}",
                            e
                        )))
                    }
                }
            }
            Err(e) => {
                tracing::warn!("FFmpeg not available: {}, falling back to raw RGB", e);
                self.encode_raw_rgb(frames, num_frames)
            }
        }
    }

    fn create_bmp_header(&self, num_frames: usize) -> Vec<u8> {
        let row_size = ((self.width * 3 + 3) & !3) as usize;
        let image_size = row_size * self.height as usize * num_frames;
        let file_size = 54 + image_size;

        let mut header = Vec::with_capacity(54);
        header.extend_from_slice(b"BM");
        header.extend_from_slice(&(file_size as u32).to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        header.extend_from_slice(&54u32.to_le_bytes());

        header.extend_from_slice(&40u32.to_le_bytes());
        header.extend_from_slice(&(self.width as i32).to_le_bytes());
        header.extend_from_slice(&(self.height as i32 * num_frames as i32).to_le_bytes());
        header.extend_from_slice(&1u16.to_le_bytes());
        header.extend_from_slice(&24u16.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&(image_size as u32).to_le_bytes());
        header.extend_from_slice(&2835u32.to_le_bytes());
        header.extend_from_slice(&2835u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());

        header
    }

    pub fn save_to_file(&self, data: &[u8], path: &Path) -> Result<(), EncodeError> {
        let mut file = std::fs::File::create(path)
            .map_err(|e| EncodeError::IoError(format!("Failed to create file: {}", e)))?;

        file.write_all(data)
            .map_err(|e| EncodeError::IoError(format!("Failed to write file: {}", e)))?;

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("FFmpeg error: {0}")]
    FFmpegError(String),

    #[error("Encoding failed: {0}")]
    EncodingFailed(String),
}

pub fn encode_frames_to_video(
    frames: &[u8],
    width: u32,
    height: u32,
    fps: u32,
    num_frames: usize,
    output_path: &Path,
    codec: VideoCodec,
) -> Result<(), EncodeError> {
    let encoder = VideoEncoder::new(width, height, fps).with_codec(codec);
    let encoded = encoder.encode_frames(frames, num_frames)?;
    encoder.save_to_file(&encoded, output_path)?;
    Ok(())
}

pub fn frames_to_raw_video(frames: &[u8], width: u32, height: u32, fps: u32) -> Vec<u8> {
    let frame_size = (width * height * 3) as usize;
    let mut output = Vec::with_capacity(frame_size * (frames.len() / frame_size));

    output.extend_from_slice(frames);

    output
}

pub fn create_video_from_tensor(
    frames: &crate::libcore::tensor::Tensor,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<Vec<u8>, EncodeError> {
    let shape = frames.shape();
    let data = match frames.data() {
        crate::libcore::tensor::TensorData::F32(arr) => arr.clone(),
        _ => {
            return Err(EncodeError::InvalidInput(
                "Only F32 tensors supported".into(),
            ))
        }
    };

    let mut rgb_data = Vec::with_capacity(data.len());
    for &value in &data {
        let scaled = ((value.clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push(scaled);
    }

    let frame_size = (width * height * 3) as usize;
    let num_frames = rgb_data.len() / frame_size;

    let encoder = VideoEncoder::new(width, height, fps);
    encoder.encode_frames(&rgb_data, num_frames)
}
