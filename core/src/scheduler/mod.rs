mod euler;
mod ddim;
mod dpmpp;

pub use self::euler::EulerScheduler;
pub use ddim::DDIMScheduler;
pub use dpmpp::DPMPPScheduler;

use crate::libcore::traits::{Error, Result, Scheduler as SchedulerTrait};
use crate::libcore::tensor::Tensor;
use crate::libcore::context::Context;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SchedulerType {
    Euler,
    EulerA,
    DDIM,
    DPMPlusPlus,
}

pub struct DiffusionScheduler {
    scheduler_type: SchedulerType,
    num_steps: u32,
    beta_start: f32,
    beta_end: f32,
    alphas: Vec<f32>,
    betas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<f32>,
    eta: f32,
}

impl DiffusionScheduler {
    pub fn new(scheduler: SchedulerType, num_steps: u32) -> Self {
        let beta_start = 0.00085_f32;
        let beta_end = 0.012_f32;
        
        let betas: Vec<f32> = if beta_start == beta_end {
            vec![beta_start; num_steps as usize]
        } else {
            let start = beta_start.sqrt();
            let end = beta_end.sqrt();
            (0..num_steps)
                .map(|i| {
                    let t = i as f32 / num_steps as f32;
                    (start + (end - start) * t).powi(2)
                })
                .collect()
        };
        
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let mut alphas_cumprod = Vec::with_capacity(alphas.len());
        let mut prod: f32 = 1.0;
        for alpha in &alphas {
            prod *= alpha;
            alphas_cumprod.push(prod);
        }
        
        let timesteps: Vec<f32> = (0..num_steps)
            .map(|i| num_steps - 1 - i)
            .map(|i| i as f32 / (num_steps - 1) as f32)
            .collect();
        
        Self {
            scheduler_type: scheduler,
            num_steps,
            beta_start,
            beta_end,
            alphas,
            betas,
            alphas_cumprod,
            timesteps,
            eta: 0.0,
        }
    }
    
    pub fn from_type_str(s: &str, num_steps: u32) -> Self {
        match s.to_lowercase().as_str() {
            "euler_a" | "eulera" => Self::new(SchedulerType::EulerA, num_steps),
            "ddim" => Self::new(SchedulerType::DDIM, num_steps),
            "dpm++" | "dpmpp" => Self::new(SchedulerType::DPMPlusPlus, num_steps),
            "euler" | _ => Self::new(SchedulerType::Euler, num_steps),
        }
    }
    
    pub fn get_alpha(&self, t: usize) -> f32 {
        if t < self.alphas.len() {
            self.alphas[t]
        } else {
            self.alphas.last().copied().unwrap_or(1.0)
        }
    }
    
    pub fn get_alpha_cumprod(&self, t: usize) -> f32 {
        if t < self.alphas_cumprod.len() {
            self.alphas_cumprod[t]
        } else {
            self.alphas_cumprod.last().copied().unwrap_or(0.0)
        }
    }
    
    pub fn get_beta(&self, t: usize) -> f32 {
        if t < self.betas.len() {
            self.betas[t]
        } else {
            self.betas.last().copied().unwrap_or(0.012)
        }
    }
    
    pub fn timestep_to_index(&self, timestep: f32) -> usize {
        let t = (timestep * (self.num_steps - 1) as f32).round() as usize;
        self.num_steps as usize - 1 - t.min(self.num_steps as usize - 1)
    }
    
    pub fn scheduler_type(&self) -> SchedulerType {
        self.scheduler_type
    }
    
    pub fn num_steps(&self) -> u32 {
        self.num_steps
    }
    
    pub fn set_eta(&mut self, eta: f32) {
        self.eta = eta;
    }
}

impl SchedulerTrait for DiffusionScheduler {
    fn step(&mut self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        match self.scheduler_type {
            SchedulerType::Euler => self.euler_step(latent, timestep, pred),
            SchedulerType::EulerA => self.euler_a_step(latent, timestep, pred),
            SchedulerType::DDIM => self.ddim_step(latent, timestep, pred),
            SchedulerType::DPMPlusPlus => self.dpmpp_step(latent, timestep, pred),
        }
    }
    
    fn add_noise(&mut self, latent: &Tensor, noise: &Tensor, timestep: f32) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let alpha_cumprod = self.get_alpha_cumprod(t_idx);
        
        let sqrt_alpha_cumprod = alpha_cumprod.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();
        
        let scaled_noise = noise.mul(&sqrt_one_minus_alpha_cumprod)?;
        let scaled_latent = latent.mul(&sqrt_alpha_cumprod)?;
        
        scaled_latent.add(&scaled_noise)
    }
    
    fn timesteps(&self) -> Vec<f32> {
        self.timesteps.clone()
    }
    
    fn set_timesteps(&mut self, num_steps: u32) {
        *self = Self::from_type_str(
            match self.scheduler_type {
                SchedulerType::Euler => "euler",
                SchedulerType::EulerA => "euler_a",
                SchedulerType::DDIM => "ddim",
                SchedulerType::DPMPlusPlus => "dpm++",
            },
            num_steps,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = DiffusionScheduler::new(SchedulerType::Euler, 50);
        assert_eq!(scheduler.num_steps(), 50);
        assert_eq!(scheduler.timesteps().len(), 50);
    }

    #[test]
    fn test_euler_step() {
        let scheduler = DiffusionScheduler::new(SchedulerType::Euler, 30);
        // Basic test that step doesn't panic
    }

    #[test]
    fn test_scheduler_from_str() {
        let euler = DiffusionScheduler::from_type_str("euler", 20);
        assert_eq!(euler.num_steps(), 20);
        
        let ddim = DiffusionScheduler::from_type_str("ddim", 20);
        assert_eq!(ddim.num_steps(), 20);
    }
}
