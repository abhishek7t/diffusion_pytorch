from torch.nn import functional as F
import torch
import numpy as np


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_like(shape, noise_fn=torch.randn, repeat=False, dtype=torch.float32):
    if repeat:
        noise = noise_fn(1, *shape[1:], dtype=dtype)
        return noise.repeat(shape[0], 1, 1, 1)
    else:
        return noise_fn(*shape, dtype=dtype)
    

class GaussianDiffusion:
    """
    Contains utilities for the diffusion model.
    """

    def __init__(self, betas, loss_type, torch_dtype) -> None:
        self.loss_type = loss_type

        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64) # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.betas = torch.tensor(betas, requires_grad=False)
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt( 1 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - self.alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.index_select(a, 0, t)
        assert out.shape == torch.Size([bs])
        return out.view(bs, *((len(x_shape) - 1) * [1]))
    
    def q_mean_variance(self, x_start, t):
        # q(x_{t} | x_0)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def q_sample(self, x_start, t, noise=None):
        '''
        Diffuse the data (t == 0 means diffused for 1 step)
        '''
        if noise is None:
            noise = torch.randn(x_start.shape)
        assert noise.shape == x_start.shape
        # mean + sqrt(var) * noise
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(1 - self.alphas_cumprod, t, x_start.shape) * noise
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        # x_0 = x_t / sqrt_alphas_cumprod - noise * sqrt(var) / sqrt_alphas_cumprod 
        assert x_t.shape == noise.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_losses(self, denoise_fn, x_start, t, noise=None):
        """
        Training loss calculation
        """
        B, C, H, W = x_start.shape.as_list()
        assert t.shape == [B]

        if noise is None:
            noise = torch.randn(x_start.shape, dtype=x_start.dtype)
        assert noise.shape == x_start.shape
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t)
        assert x_noisy.shape == x_start.shape
        assert x_recon.shape[:3] == [B, C, H] and len(x_recon.shape) == 4

        if self.loss_type == 'noisepred':
            # prdict the noise instead of x_start. Seems to be weighted naturally like SNR
            assert x_recon.shape == x_start.shape
            # Compute element-wise squared difference
            loss = (x_recon - noise) ** 2
            # Take mean over all dimensions except the first (batch dimension)
            losses = loss.mean(dim=list(range(1, loss.ndim)))
        else:
            raise NotImplementedError(self.loss_type)
        
        assert loss.shape == [B]
        return losses
    
    def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised):
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        else:
            raise NotImplementedError(self.loss_type)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        assert model_mean.shape == x_recon.shape == x.shape
        assert posterior_variance.shape == posterior_log_variance.shape == [x.shape[0], 1, 1, 1]
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t,
                                                                 clip_denoised=clip_denoised)
        noise = noise_like(x.shape, noise_fn, repeat_noise)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        return model_mean + torch.exp(.5 * model_log_variance) * noise
    
    def p_sample_loop(self, denoise_fn, *, shape, noise_fn=torch.randn):
        """
        Generate samples
        """
        i_0 = torch.tensor(self.num_timesteps - 1, dtype=torch.int32)
        assert isinstance(shape, (tuple, list))
        img_0 = noise_fn(size=shape, dtype=torch.float32)
        img_ = img_0
        for i_ in range(i_0, -1, -1):
            img_ = self.p_sample(denoise_fn=denoise_fn, x=img_,
                                 t=torch.full((shape[0],), i_, dtype=torch.int32),  # Create a tensor of timesteps
                                 noise_fn=noise_fn)
            
        assert img_.shape == torch.Size(shape)
        return img_
    
    def p_sample_loop_trajectory(self, denoise_fn, *, shape, noise_fn=torch.randn, repeat_noise_steps=-1):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
            repeat_noise_steps (int): Number od denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elements.
        """
        i_0 = torch.tensor(self.num_timesteps - 1, dtype=torch.int32)
        assert isinstance(shape, (tuple, list))
        img_0 = noise_fn(size=shape, dtype=torch.float32)
        times = [torch.tensor(i_0)]
        imgs = [img_0]
        
        # Steps with repeated noise
        for i_ in range(i_0, i_0 - repeat_noise_steps, -1):
            img_ = self.p_sample(denoise_fn=denoise_fn, x=img_,
                                 t=torch.full((shape[0],), i_, dtype=torch.int32),  # Create a tensor of timesteps
                                 noise_fn=noise_fn, repeat_noise=True)
            imgs.append(img_)
            times.append(i_)

        # Steps with different noise for each batch element
        for i_ in range(i_0 - repeat_noise_steps, -1, -1):
            img_ = self.p_sample(denoise_fn=denoise_fn, x=img_,
                                 t=torch.full((shape[0],), i_, dtype=torch.int32),  # Create a tensor of timesteps
                                 noise_fn=noise_fn, repeat_noise=False)
            imgs.append(img_)
            times.append(i_)

        # Convert list of tensors to a single tensor along the 0th dimension (time axis)
        times = torch.stack(times)
        imgs = torch.stack(imgs)

        assert imgs[-1].shape == torch.Size(shape)
        return times, imgs
    
    def interpolate(aelf, denoise_fn, *, shape, noise_fn=torch.randn):
        """
        Interpolate between images.
        t == 0 means diffuse images for 1 timestep before mixing.
        """
        assert isinstance(shape, (tuple, list))