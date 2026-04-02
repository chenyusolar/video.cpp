use crate::libcore::tensor::{Tensor, TensorData};
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

        let sqrt_one_minus_alpha_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_one_minus_alpha_cumprod),
        );
        let sqrt_alpha_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_alpha_cumprod),
        );

        let pred_original = if sqrt_one_minus_alpha_cumprod > 0.0 {
            latent
                .sub(&pred.mul(&sqrt_one_minus_alpha_tensor)?)?
                .div(&sqrt_alpha_tensor)?
        } else {
            pred.clone()
        };

        let sqrt_alpha_cumprod_prev = alpha_cumprod_prev.sqrt();
        let sqrt_one_minus_alpha_cumprod_prev = (1.0 - alpha_cumprod_prev).sqrt();

        let sqrt_alpha_prev_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_alpha_cumprod_prev),
        );
        let sqrt_one_minus_prev_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_one_minus_alpha_cumprod_prev),
        );

        let pred_scaled = pred_original.mul(&sqrt_alpha_prev_tensor)?;
        let noise_scaled = pred.mul(&sqrt_one_minus_prev_tensor)?;

        if self.eta > 0.0 {
            let sigma = self.eta
                * ((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t)
                    / (1.0 - alpha_cumprod_prev))
                    .sqrt();
            let sigma_tensor =
                Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(sigma));
            let noise = latent.randn_like()?;
            let sigma_term = noise.mul(&sigma_tensor)?;
            pred_scaled.add(&sigma_term)?.add(&noise_scaled)
        } else {
            pred_scaled.add(&noise_scaled)
        }
    }
}
