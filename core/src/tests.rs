#[cfg(test)]
mod tests {
    use video_core::libcore::tensor::{Tensor, TensorShape, TensorData, DType};
    use video_core::libcore::traits::Backend;
    use video_core::backend::CpuBackend;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = TensorShape::new(vec![2, 2]);
        let tensor = Tensor::from_data(shape, TensorData::F32(data.clone()));
        
        assert_eq!(tensor.volume(), 4);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_add() {
        let shape = TensorShape::new(vec![2, 2]);
        let a = Tensor::from_data(shape.clone(), TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]));
        let b = Tensor::from_data(shape.clone(), TensorData::F32(vec![1.0, 1.0, 1.0, 1.0]));
        
        let result = a.add(&b).unwrap();
        
        match result.data() {
            TensorData::F32(data) => {
                assert_eq!(data, &[2.0, 3.0, 4.0, 5.0]);
            }
            _ => panic!("Expected F32 data"),
        }
    }

    #[test]
    fn test_tensor_mul() {
        let shape = TensorShape::new(vec![2, 2]);
        let a = Tensor::from_data(shape.clone(), TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]));
        let b = Tensor::from_data(shape.clone(), TensorData::F32(vec![2.0, 2.0, 2.0, 2.0]));
        
        let result = a.mul(&b).unwrap();
        
        match result.data() {
            TensorData::F32(data) => {
                assert_eq!(data, &[2.0, 4.0, 6.0, 8.0]);
            }
            _ => panic!("Expected F32 data"),
        }
    }

    #[test]
    fn test_cpu_backend.randn() {
        let backend = CpuBackend::new();
        let shape = TensorShape::new(vec![2, 3, 4]);
        
        let tensor = backend.randn(shape.clone()).unwrap();
        
        assert_eq!(tensor.volume(), 24);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_cpu_backend_operations() {
        let backend = CpuBackend::new();
        
        let a = backend.randn(TensorShape::new(vec![10, 10])).unwrap();
        let b = backend.randn(TensorShape::new(vec![10, 10])).unwrap();
        
        let sum = backend.tensor_add(&a, &b).unwrap();
        let product = backend.tensor_mul(&a, &b).unwrap();
        
        assert_eq!(sum.volume(), 100);
        assert_eq!(product.volume(), 100);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_data(
            TensorShape::new(vec![2, 4]),
            TensorData::F32(data),
        );
        
        let reshaped = tensor.reshape(&[4, 2]).unwrap();
        assert_eq!(reshaped.volume(), 8);
    }

    #[test]
    fn test_scheduler_euler() {
        use video_core::scheduler::{DiffusionScheduler, SchedulerType};
        
        let scheduler = DiffusionScheduler::new(SchedulerType::Euler, 30);
        let timesteps = scheduler.timesteps();
        
        assert_eq!(timesteps.len(), 30);
    }

    #[test]
    fn test_scheduler_ddim() {
        use video_core::scheduler::{DiffusionScheduler, SchedulerType};
        
        let scheduler = DiffusionScheduler::new(SchedulerType::DDIM, 20);
        let timesteps = scheduler.timesteps();
        
        assert_eq!(timesteps.len(), 20);
    }

    #[test]
    fn test_config_from_env() {
        use video_core::config::Config;
        
        std::env::set_var("VIDEO_BACKEND", "cpu");
        std::env::set_var("VIDEO_STEPS", "50");
        
        let config = Config::from_env();
        
        assert_eq!(config.generation.steps, 50);
    }

    #[test]
    fn test_video_encoder() {
        use video_core::encoder::{VideoEncoder, VideoCodec};
        
        let encoder = VideoEncoder::new(512, 512, 24);
        
        let frame_size = 512 * 512 * 3;
        let frames = vec![0u8; frame_size];
        
        let result = encoder.encode_frames(&frames, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_loader_gguf() {
        use video_core::model::gguf::{GGUFFile, GGUFDType};
        
        let dtype = GGUFDType::from_u32(10);
        assert!(matches!(dtype, GGUFDType::Q4_K));
    }
}
