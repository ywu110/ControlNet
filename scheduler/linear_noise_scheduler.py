import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        """
        Initialize the LinearNoiseScheduler.
        
        x_t = sqrt(alpha_t...alpha_0) * x_0 + sqrt(1 - alpha_t...alpha_0) * \epsilon
        where alpha_t = 1 - beta_t, and beta_t is a linear schedule from beta_start to beta_end. 
        \epsilon is the noise added at each timestep. It is sampled from a standard normal distribution.

        Args:
            num_timesteps (int): Number of timesteps.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps) # beta_0, beta_1, ..., beta_T
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # alpha_0, alpha_0*alpha_1, ..., alpha_0*alpha_1*...*alpha_T
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # sqrt(alpha_0), sqrt(alpha_0*alpha_1), ..., sqrt(alpha_0*alpha_1*...*alpha_T)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod) # sqrt(1 - alpha_0), sqrt(1 - alpha_0*alpha_1), ..., sqrt(1 - alpha_0*alpha_1*...*alpha_T)
    
    
    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, t: int):
        """
        Forward Process: Add noise to the original image at timestep t.
        x_t = sqrt(alpha_t...alpha_0) * x_0 + sqrt(1 - alpha_t...alpha_0) * \epsilon
        
        Args:
            original (torch.Tensor): Original image.
            noise (torch.Tensor): Noise to be added.
            t (int): Timestep.
        
        Returns:
            torch.Tensor: Noisy image.
        """
        
        batch_size = original.shape[0] 
        
        sqrt_alpha_cum_prod = self.sqrt_alphas_cumprod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alphas_cumprod.to(original.device)[t].reshape(batch_size)
        
        # reshape to match the original tensor dimensions: from (batch_size,) to (batch_size, 1, 1, 1)
        for _ in range(len(original.shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        # Apply Forward process equation: 
        # x_t = sqrt(alpha_t...alpha_0) * x_0 + sqrt(1 - alpha_t...alpha_0) * \epsilon
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
    
    
    def sample_prev_timestep(self, xt: torch.Tensor, noise_pred: torch.Tensor, t: int):
        
        # xt = \sqrt(\hat{\alpha_t}) * x_0 + \sqrt{1 - \hat{\alpha_t}} * \epsilon ==> We use network to predict \epsilon (noise_pred)
        # x_0 = \frac{1}{\sqrt{\hat{\alpha_t}}} * (xt - \sqrt{1 - \hat{\alpha_t}} * noise_pred)
        x0 = xt - (self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t] * noise_pred)
        x0 = x0 / self.sqrt_alphas_cumprod.to(xt.device)[t]
        
        x0 = x0.clamp(-1, 1) # Clamp the values to be in the range [-1, 1] to avoid numerical instability
        
        # /mu_\theta(x_t, t) = \frac{1}{\sqrt{\hat{\alpha_t}}} * (xt - beta_t * noise_pred / sqrt(1 - \hat{\alpha_t}))
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            # variance = \frac{1 - \hat{\alpha_{t-1}}}{1 - \hat{\alpha_t}} * beta_t
            variance = (1 - self.alphas_cumprod.to(xt.device)[t - 1]) / (1.0 - self.alphas_cumprod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            
            z = torch.randn(xt.shape).to(xt.device)
            
            # Sample from the normal distribution
            return mean + sigma * z, x0