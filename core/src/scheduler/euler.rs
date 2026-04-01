use crate::libcore::tensor::Tensor;
use crate::libcore::traits::Result;

use super::DiffusionScheduler;

impl DiffusionScheduler {
    pub fn euler_step(&self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let dt = 1.0 / self.num_steps as f32;

        let alpha = self.get_alpha(t_idx);
        let beta = self.get_beta(t_idx);

        let ddim_deterministic = if self.eta == 0.0 { 1.0 } else { 0.0 };

        let pred_scaled = pred.mul(&beta)?;
        latent.sub(&pred_scaled)
    }

    pub fn euler_a_step(&self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let dt = 1.0 / self.num_steps as f32;

        let alpha = self.get_alpha(t_idx);
        let alpha_cumprod = self.get_alpha_cumprod(t_idx);

        let pred_x0 = if alpha_cumprod > 0.0 {
            let sqrt_alpha = alpha.sqrt();
            let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();

            if sqrt_one_minus_alpha_cumprod > 0.0 {
                let pred_original = latent.sub(&pred.mul(&sqrt_one_minus_alpha_cumprod)?)?;
                pred_original.div(&sqrt_alpha)?
            } else {
                pred.clone()
            }
        } else {
            pred.clone()
        };

        let prev_t_idx = if t_idx > 0 { t_idx - 1 } else { 0 };
        let prev_alpha_cumprod = self.get_alpha_cumprod(prev_t_idx);
        let sqrt_prev_alpha_cumprod = prev_alpha_cumprod.sqrt();
        let sqrt_one_minus_prev_alpha_cumprod = (1.0 - prev_alpha_cumprod).sqrt();

        let noise = pred.mul(&sqrt_one_minus_prev_alpha_cumprod)?;
        pred_x0.mul(&sqrt_prev_alpha_cumprod)?.add(&noise)
    }
}
