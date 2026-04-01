use crate::libcore::tensor::Tensor;
use crate::libcore::traits::Result;

use super::DiffusionScheduler;

impl DiffusionScheduler {
    pub fn ddim_step(&mut self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let prev_t_idx = if t_idx > 0 { t_idx - 1 } else { 0 };

        let alpha_cumprod_t = self.get_alpha_cumprod(t_idx);
        let alpha_cumprod_prev = self.get_alpha_cumprod(prev_t_idx);

        let sqrt_alpha_cumprod = alpha_cumprod_t.sqrt();
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod_t).sqrt();

        let pred_original = if sqrt_one_minus_alpha_cumprod > 0.0 {
            latent
                .sub(&pred.mul(&sqrt_one_minus_alpha_cumprod)?)?
                .div(&sqrt_alpha_cumprod)?
        } else {
            pred.clone()
        };

        let sqrt_alpha_cumprod_prev = alpha_cumprod_prev.sqrt();
        let sqrt_one_minus_alpha_cumprod_prev = (1.0 - alpha_cumprod_prev).sqrt();

        let pred_scaled = pred_original.mul(&sqrt_alpha_cumprod_prev)?;
        let noise_scaled = pred.mul(&sqrt_one_minus_alpha_cumprod_prev)?;

        if self.eta > 0.0 {
            let sigma = self.eta
                * ((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t)
                    / (1.0 - alpha_cumprod_prev))
                    .sqrt();
            let noise = latent.randn_like()?;
            let sigma_term = noise.mul(&sigma)?;
            pred_scaled.add(&sigma_term)?.add(&noise_scaled)
        } else {
            pred_scaled.add(&noise_scaled)
        }
    }
}
