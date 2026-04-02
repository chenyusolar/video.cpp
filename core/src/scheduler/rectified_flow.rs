use crate::libcore::tensor::{Tensor, TensorData, TensorShape};
use crate::libcore::traits::{Error, Result, Scheduler as SchedulerTrait};

#[derive(Debug, Clone)]
pub struct RectifiedFlowScheduler {
    pub num_train_timesteps: u32,
    pub num_inference_steps: Option<u32>,
    pub timesteps: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub shift: Option<f32>,
    pub shifting: Option<String>,
    pub base_resolution: u32,
    pub target_shift_terminal: Option<f32>,
    pub sampler: String,
}

impl RectifiedFlowScheduler {
    pub fn new() -> Self {
        Self {
            num_train_timesteps: 1000,
            num_inference_steps: None,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            shift: None,
            shifting: None,
            base_resolution: 32 * 32,
            target_shift_terminal: None,
            sampler: "Uniform".to_string(),
        }
    }

    pub fn with_sampler(mut self, sampler: &str) -> Self {
        self.sampler = sampler.to_string();
        self
    }

    pub fn with_shift(mut self, shift: f32) -> Self {
        self.shift = Some(shift);
        self
    }

    pub fn with_shifting(mut self, shifting: &str) -> Self {
        self.shifting = Some(shifting.to_string());
        self
    }

    pub fn with_num_train_timesteps(mut self, num_timesteps: u32) -> Self {
        self.num_train_timesteps = num_timesteps;
        self
    }

    pub fn set_timesteps(&mut self, num_inference_steps: u32, sample_shape: Option<TensorShape>) {
        self.num_inference_steps = Some(num_inference_steps);

        let timesteps: Vec<f32> = match self.sampler.as_str() {
            "Uniform" => (0..num_inference_steps)
                .map(|i| {
                    let t = i as f32 / num_inference_steps as f32;
                    1.0 - t
                })
                .collect(),
            "LinearQuadratic" => self.linear_quadratic_schedule(num_inference_steps),
            "Constant" => {
                let shift = self.shift.unwrap_or(1.0);
                (0..num_inference_steps)
                    .map(|i| {
                        let t = i as f32 / num_inference_steps as f32;
                        Self::time_shift(shift, 1.0, t)
                    })
                    .collect()
            }
            _ => (0..num_inference_steps)
                .map(|i| {
                    let t = i as f32 / num_inference_steps as f32;
                    1.0 - t
                })
                .collect(),
        };

        if let Some(ref shifting) = self.shifting {
            let shape = sample_shape.unwrap_or_else(|| TensorShape::new(vec![1, 32 * 32]));
            self.timesteps = self.shift_timesteps(&shape, &timesteps, shifting);
        } else {
            self.timesteps = timesteps;
        }

        self.sigmas = self.timesteps.clone();
    }

    fn shift_timesteps(
        &self,
        sample_shape: &TensorShape,
        timesteps: &[f32],
        shifting: &str,
    ) -> Vec<f32> {
        match shifting {
            "SD3" => self.sd3_resolution_dependent_timestep_shift(sample_shape, timesteps),
            "SimpleDiffusion" => {
                self.simple_diffusion_resolution_dependent_timestep_shift(sample_shape, timesteps)
            }
            _ => timesteps.to_vec(),
        }
    }

    fn sd3_resolution_dependent_timestep_shift(
        &self,
        sample_shape: &TensorShape,
        timesteps: &[f32],
    ) -> Vec<f32> {
        let dims = sample_shape.dims();
        let m = if dims.len() >= 3 {
            dims.iter().skip(2).product::<u32>() as f32
        } else {
            dims.iter().product::<u32>() as f32
        };

        let n_tokens = self.base_resolution as f32;
        let shift = Self::get_normal_shift(m as u32, n_tokens as u32, 4096);

        timesteps
            .iter()
            .map(|&t| Self::time_shift(shift, 1.0, t))
            .collect()
    }

    fn simple_diffusion_resolution_dependent_timestep_shift(
        &self,
        sample_shape: &TensorShape,
        timesteps: &[f32],
    ) -> Vec<f32> {
        let n = 32 * 32;
        let dims = sample_shape.dims();
        let m = if dims.len() >= 3 {
            dims.iter().skip(2).product::<u32>() as f32
        } else {
            dims.iter().product::<u32>() as f32
        };

        let snr: Vec<f32> = timesteps
            .iter()
            .map(|&t| {
                let snr = (t / (1.0 - t)).powi(2);
                let shift_snr =
                    snr.log(std::f32::consts::E) + 2.0 * (m / n as f32).log(std::f32::consts::E);
                1.0 / (1.0 + (-shift_snr).exp())
            })
            .collect();

        snr
    }

    fn time_shift(mu: f32, sigma: f32, t: f32) -> f32 {
        let exp_term = mu.exp();
        let denominator = (1.0 / t - 1.0).powi(2);
        exp_term / (exp_term + denominator.powf(sigma))
    }

    fn get_normal_shift(n_tokens: u32, min_tokens: u32, max_tokens: u32) -> f32 {
        let min_shift = 0.95;
        let max_shift = 2.05;
        let m = (max_shift - min_shift) / (max_tokens as f32 - min_tokens as f32);
        let b = min_shift - m * min_tokens as f32;
        m * n_tokens as f32 + b
    }

    fn linear_quadratic_schedule(&self, num_steps: u32) -> Vec<f32> {
        let threshold_noise = 0.025_f32;
        let linear_steps = num_steps / 2;

        let mut schedule: Vec<f32> = Vec::with_capacity(num_steps as usize);

        for i in 0..linear_steps {
            let t = i as f32 / linear_steps as f32;
            schedule.push(t * threshold_noise);
        }

        let threshold_noise_step_diff = linear_steps as f32 - threshold_noise * num_steps as f32;
        let quadratic_steps = num_steps - linear_steps;
        let quadratic_coef =
            threshold_noise_step_diff / (linear_steps as f32 * (quadratic_steps as f32).powi(2));
        let linear_coef = threshold_noise / linear_steps as f32
            - 2.0 * threshold_noise_step_diff / ((quadratic_steps as f32).powi(2));
        let const_term = quadratic_coef * (linear_steps as f32).powi(2);

        for i in linear_steps..num_steps {
            let t = i as f32;
            let quad = quadratic_coef * t.powi(2) + linear_coef * t + const_term;
            schedule.push(quad);
        }

        let mut full_schedule: Vec<f32> = schedule.iter().map(|x| 1.0 - x).collect();
        full_schedule.push(1.0);

        while full_schedule.len() > num_steps as usize {
            full_schedule.pop();
        }

        full_schedule
    }

    pub fn timesteps(&self) -> Vec<f32> {
        self.timesteps.clone()
    }

    pub fn get_initial_timesteps(&self, num_inference_steps: u32) -> Vec<f32> {
        match self.sampler.as_str() {
            "Uniform" => (0..num_inference_steps)
                .map(|i| 1.0 - i as f32 / num_inference_steps as f32)
                .collect(),
            "LinearQuadratic" => self.linear_quadratic_schedule(num_inference_steps),
            "Constant" => {
                let shift = self.shift.unwrap_or(1.0);
                (0..num_inference_steps)
                    .map(|i| {
                        let t = 1.0 - i as f32 / num_inference_steps as f32;
                        Self::time_shift(shift, 1.0, t)
                    })
                    .collect()
            }
            _ => (0..num_inference_steps)
                .map(|i| 1.0 - i as f32 / num_inference_steps as f32)
                .collect(),
        }
    }
}

impl Default for RectifiedFlowScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulerTrait for RectifiedFlowScheduler {
    fn step(&mut self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        if self.timesteps.is_empty() {
            return Err(Error::Model("Timesteps not set".into()));
        }

        let timestep_idx = self
            .timesteps
            .iter()
            .position(|&t| (t - timestep).abs() < 1e-6)
            .or_else(|| self.timesteps.iter().position(|&t| t < timestep))
            .unwrap_or(0);

        let dt = if timestep_idx + 1 < self.timesteps.len() {
            self.timesteps[timestep_idx] - self.timesteps[timestep_idx + 1]
        } else {
            self.timesteps[timestep_idx]
        };

        let pred_data = pred.data();
        let latent_data = latent.data();

        let scale_factor = 1.0 - dt;

        let new_data: Vec<f32> = match (pred_data, latent_data) {
            (TensorData::F32(pred), TensorData::F32(latent)) => latent
                .iter()
                .zip(pred.iter())
                .map(|(l, p)| l - scale_factor * p)
                .collect(),
            (TensorData::F32(pred), TensorData::F32Scalar(s)) => {
                pred.iter().map(|p| s - scale_factor * p).collect()
            }
            _ => {
                return Err(Error::Model("Unsupported tensor data types".into()));
            }
        };

        Ok(Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32(new_data),
        ))
    }

    fn add_noise(&mut self, original: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor> {
        let sigmas = self.sigmas.clone();

        let sigma = sigmas
            .iter()
            .min_by(|a, b| {
                (*a - timestep)
                    .abs()
                    .partial_cmp(&(*b - timestep).abs())
                    .unwrap()
            })
            .copied()
            .unwrap_or(1.0);

        let alpha = 1.0 - sigma;

        let original_data = original.data();
        let noise_data = noise.data();

        let result: Vec<f32> = match (original_data, noise_data) {
            (TensorData::F32(orig), TensorData::F32(noise_vec)) => orig
                .iter()
                .zip(noise_vec.iter())
                .map(|(o, n)| alpha * o + sigma * n)
                .collect(),
            _ => {
                return Err(Error::Model(
                    "Unsupported tensor data types for add_noise".into(),
                ));
            }
        };

        Ok(Tensor::from_data(
            original.shape().clone(),
            TensorData::F32(result),
        ))
    }

    fn timesteps(&self) -> Vec<f32> {
        self.timesteps.clone()
    }

    fn set_timesteps(&mut self, num_steps: u32) {
        self.set_timesteps_internal(num_steps, None);
    }
}

impl RectifiedFlowScheduler {
    fn set_timesteps_internal(
        &mut self,
        num_inference_steps: u32,
        sample_shape: Option<TensorShape>,
    ) {
        self.num_inference_steps = Some(num_inference_steps);

        self.timesteps = self.get_initial_timesteps(num_inference_steps);

        if let Some(ref shifting) = self.shifting {
            let shape = sample_shape.unwrap_or_else(|| TensorShape::new(vec![1, 32 * 32]));
            self.timesteps = self.shift_timesteps(&shape, &self.timesteps, shifting);
        }

        self.sigmas = self.timesteps.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectified_flow_scheduler() {
        let scheduler = RectifiedFlowScheduler::new()
            .with_sampler("Uniform")
            .with_num_train_timesteps(1000);

        let mut scheduler = scheduler;
        scheduler.set_timesteps(50, None);

        assert_eq!(scheduler.timesteps().len(), 50);
    }

    #[test]
    fn test_rectified_flow_step() {
        let mut scheduler = RectifiedFlowScheduler::new().with_sampler("Uniform");
        scheduler.set_timesteps(50, None);

        let latent = Tensor::from_data(
            TensorShape::new(vec![1, 4, 32, 32]),
            TensorData::F32(vec![0.0; 1 * 4 * 32 * 32]),
        );
        let pred = Tensor::from_data(
            TensorShape::new(vec![1, 4, 32, 32]),
            TensorData::F32(vec![0.1; 1 * 4 * 32 * 32]),
        );

        let result = scheduler.step(&latent, scheduler.timesteps()[0], &pred);
        assert!(result.is_ok());
    }
}
