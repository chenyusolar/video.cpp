use crate::libcore::tensor::{Tensor, TensorData};
use crate::libcore::traits::Result;

use super::DiffusionScheduler;

impl DiffusionScheduler {
    pub fn euler_step(&self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let _dt = 1.0 / self.num_steps as f32;

        let alpha = self.get_alpha(t_idx);
        let beta = self.get_beta(t_idx);

        let beta_tensor = Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(beta));
        let alpha_tensor = Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(alpha));

        let pred_scaled = pred.mul(&beta_tensor)?;
        latent.sub(&pred_scaled)
    }

    pub fn euler_a_step(&self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let _dt = 1.0 / self.num_steps as f32;

        let alpha = self.get_alpha(t_idx);
        let alpha_cumprod = self.get_alpha_cumprod(t_idx);

        let sqrt_alpha_tensor =
            Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(alpha.sqrt()));
        let sqrt_one_minus_alpha_cumprod = (1.0 - alpha_cumprod).sqrt();
        let sqrt_one_minus_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_one_minus_alpha_cumprod),
        );

        let pred_x0 = if sqrt_one_minus_alpha_cumprod > 0.0 {
            let pred_original = latent.sub(&pred.mul(&sqrt_one_minus_tensor)?)?;
            pred_original.div(&sqrt_alpha_tensor)?
        } else {
            pred.clone()
        };

        let prev_t_idx = if t_idx > 0 { t_idx - 1 } else { 0 };
        let prev_alpha_cumprod = self.get_alpha_cumprod(prev_t_idx);
        let sqrt_prev_alpha_cumprod = prev_alpha_cumprod.sqrt();
        let sqrt_one_minus_prev_alpha_cumprod = (1.0 - prev_alpha_cumprod).sqrt();

        let sqrt_prev_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_prev_alpha_cumprod),
        );
        let sqrt_one_minus_prev_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(sqrt_one_minus_prev_alpha_cumprod),
        );

        let noise = pred.mul(&sqrt_one_minus_prev_tensor)?;
        pred_x0.mul(&sqrt_prev_tensor)?.add(&noise)
    }
}
