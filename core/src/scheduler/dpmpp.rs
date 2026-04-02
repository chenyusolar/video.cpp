use crate::libcore::tensor::{Tensor, TensorData};
use crate::libcore::traits::Result;

use super::DiffusionScheduler;

impl DiffusionScheduler {
    pub fn dpmpp_step(&self, latent: &Tensor, timestep: f32, pred: &Tensor) -> Result<Tensor> {
        let t_idx = self.timestep_to_index(timestep);
        let prev_t_idx = if t_idx > 0 { t_idx - 1 } else { 0 };

        let alpha_cumprod_t = self.get_alpha_cumprod(t_idx);
        let alpha_cumprod_prev = self.get_alpha_cumprod(prev_t_idx);

        let lambda_t = if alpha_cumprod_t > 0.0 && (1.0 - alpha_cumprod_t) > 0.0 {
            (alpha_cumprod_t / (1.0 - alpha_cumprod_t)).sqrt().atanh()
        } else {
            0.0
        };

        let lambda_s = if t_idx > 0 {
            let alpha_cumprod_s = self.get_alpha_cumprod(t_idx - 1);
            if alpha_cumprod_s > 0.0 && (1.0 - alpha_cumprod_s) > 0.0 {
                (alpha_cumprod_s / (1.0 - alpha_cumprod_s)).sqrt().atanh()
            } else {
                0.0
            }
        } else {
            lambda_t
        };

        let delta_lambda = lambda_t - lambda_s;

        let sigma_t = if alpha_cumprod_t > 0.0 {
            (1.0 - alpha_cumprod_t).sqrt()
        } else {
            1.0
        };

        let sigma_s = if t_idx > 0 && prev_t_idx < self.alphas_cumprod.len() {
            let alpha_cumprod_s = self.get_alpha_cumprod(t_idx - 1);
            if alpha_cumprod_s > 0.0 {
                (1.0 - alpha_cumprod_s).sqrt()
            } else {
                1.0
            }
        } else {
            1.0
        };

        let h = if delta_lambda.abs() > 1e-6 {
            ((sigma_t / sigma_s).powi(2) - 1.0) / delta_lambda
        } else {
            -2.0 * sigma_t
        };

        let sigma_t_tensor =
            Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(sigma_t));
        let alpha_sqrt_tensor = Tensor::from_data(
            latent.shape().clone(),
            TensorData::F32Scalar(alpha_cumprod_t.sqrt()),
        );

        let pred_x0 = if sigma_t > 0.0 && alpha_cumprod_t > 0.0 {
            latent
                .sub(&pred.mul(&sigma_t_tensor)?)?
                .div(&alpha_sqrt_tensor)?
        } else {
            pred.clone()
        };

        let sigma_s_to_t = sigma_s * (-h).exp();
        let sigma_t_to_s = sigma_t * (h).exp();

        let sigma_s_to_t_tensor =
            Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(sigma_s_to_t));
        let sigma_t_to_s_tensor =
            Tensor::from_data(latent.shape().clone(), TensorData::F32Scalar(sigma_t_to_s));

        let direction = pred.mul(&sigma_t_to_s_tensor)?;
        pred_x0.mul(&sigma_s_to_t_tensor)?.add(&direction)
    }
}
