#!/usr/bin/env python3
"""
⏰ Noise Scheduler - DDPM & DDIM
Gère le processus de diffusion (forward: ajout de bruit, backward: débruitage)
"""

import torch
import math

class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models)
    Schedule de bruit linéaire avec variance scheduling
    """
    def __init__(
        self,
        num_steps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type="linear"
    ):
        self.num_steps = num_steps
        
        # Beta schedule
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            # Cosine schedule (meilleur pour images)
            self.betas = self._cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        # Alphas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        
        # Précalculer pour efficacité
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
        # Pour le reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alpha_bars_minus_one = torch.sqrt(1.0 / self.alpha_bars - 1)
        
        # Variance pour sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule comme dans improved DDPM
        Plus smooth que linear
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_clean, timesteps, noise=None):
        """
        Forward process: q(x_t | x_0)
        Ajoute du bruit à x_clean
        
        x_clean: [batch, channels, h, w]
        timesteps: [batch] indices de timestep
        noise: [batch, channels, h, w] ou None (généré automatiquement)
        
        returns: x_noisy
        """
        if noise is None:
            noise = torch.randn_like(x_clean)
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps]
        
        # Reshape pour broadcasting
        while len(sqrt_alpha_bar.shape) < len(x_clean.shape):
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_noisy = sqrt_alpha_bar * x_clean + sqrt_one_minus_alpha_bar * noise
        
        return x_noisy
    
    def step(self, x_t, timestep, predicted_noise):
        """
        Reverse process: p(x_{t-1} | x_t)
        Un step de débruitage
        
        x_t: [batch, channels, h, w] image bruitée au timestep t
        timestep: int, timestep actuel
        predicted_noise: [batch, channels, h, w] bruit prédit par le modèle
        
        returns: x_{t-1}, predicted x_0
        """
        # Récupérer les valeurs pour ce timestep
        alpha = self.alphas[timestep]
        alpha_bar = self.alpha_bars[timestep]
        beta = self.betas[timestep]
        
        if timestep > 0:
            alpha_bar_prev = self.alpha_bars_prev[timestep]
            posterior_variance = self.posterior_variance[timestep]
        else:
            alpha_bar_prev = torch.tensor(1.0)
            posterior_variance = torch.tensor(0.0)
        
        # Prédire x_0 depuis x_t et le bruit
        sqrt_recip_alpha_bar = 1.0 / torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        
        pred_x0 = sqrt_recip_alpha_bar * (x_t - sqrt_one_minus_alpha_bar * predicted_noise)
        
        # Calculer mean de q(x_{t-1} | x_t, x_0)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
        
        # Formule DDPM
        coef1 = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        coef2 = torch.sqrt(alpha_bar_prev) * beta / (1.0 - alpha_bar)
        
        mean = coef1 * x_t + coef2 * pred_x0
        
        # Ajouter du bruit si pas le dernier step
        if timestep > 0:
            noise = torch.randn_like(x_t)
            x_prev = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = mean
        
        return x_prev, pred_x0

class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models)
    Plus rapide que DDPM (peut skip des steps)
    Déterministe (pas de bruit ajouté)
    """
    def __init__(
        self,
        num_train_steps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        self.num_train_steps = num_train_steps
        
        # Beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, num_train_steps)
        
        # Alphas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Pour efficacité
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
    
    def add_noise(self, x_clean, timesteps, noise=None):
        """Identique à DDPM"""
        if noise is None:
            noise = torch.randn_like(x_clean)
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps]
        
        while len(sqrt_alpha_bar.shape) < len(x_clean.shape):
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        x_noisy = sqrt_alpha_bar * x_clean + sqrt_one_minus_alpha_bar * noise
        return x_noisy
    
    def step(self, x_t, timestep, next_timestep, predicted_noise, eta=0.0):
        """
        DDIM step (peut skip des timesteps)
        
        x_t: image bruitée au timestep t
        timestep: timestep actuel
        next_timestep: timestep suivant (peut être t-1, t-10, etc.)
        predicted_noise: bruit prédit
        eta: 0=déterministe, 1=stochastique comme DDPM
        
        returns: x_{next_timestep}
        """
        alpha_bar_t = self.alpha_bars[timestep]
        
        if next_timestep >= 0:
            alpha_bar_next = self.alpha_bars[next_timestep]
        else:
            alpha_bar_next = torch.tensor(1.0)
        
        # Prédire x_0
        sqrt_recip_alpha_bar = 1.0 / torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        
        pred_x0 = sqrt_recip_alpha_bar * (x_t - sqrt_one_minus_alpha_bar * predicted_noise)
        
        # Direction pointant vers x_t
        sqrt_one_minus_alpha_bar_next = torch.sqrt(1.0 - alpha_bar_next)
        dir_xt = sqrt_one_minus_alpha_bar_next * predicted_noise
        
        # x_{t-1}
        sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
        x_next = sqrt_alpha_bar_next * pred_x0 + dir_xt
        
        # Optionnel: ajouter du bruit stochastique
        if eta > 0 and next_timestep >= 0:
            variance = eta * torch.sqrt(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * 
                (1.0 - alpha_bar_t / alpha_bar_next)
            )
            noise = torch.randn_like(x_t)
            x_next = x_next + variance * noise
        
        return x_next, pred_x0
    
    def get_sampling_timesteps(self, num_inference_steps):
        """
        Créer une séquence de timesteps pour sampling
        Permet de skip des steps (ex: 1000 steps → 50 steps)
        """
        step = self.num_train_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_train_steps, step)
        timesteps = timesteps.flip(0)  # Partir de T vers 0
        return timesteps

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("⏰ NOISE SCHEDULER TEST")
    print("="*60)
    
    # Test DDPM
    print("\n📊 DDPM Scheduler")
    ddpm = DDPMScheduler(num_steps=1000, schedule_type="cosine")
    
    # Forward process (ajout de bruit)
    x_clean = torch.randn(2, 4, 64, 64).to(device)
    timesteps = torch.tensor([500, 800]).to(device)
    noise = torch.randn_like(x_clean)
    
    x_noisy = ddpm.add_noise(x_clean, timesteps, noise)
    print(f"   Clean shape: {x_clean.shape}")
    print(f"   Noisy shape: {x_noisy.shape}")
    
    # Reverse process (un step)
    predicted_noise = torch.randn_like(x_noisy)
    x_prev, pred_x0 = ddpm.step(x_noisy, timesteps[0].item(), predicted_noise[0:1])
    print(f"   Denoised shape: {x_prev.shape}")
    
    # Test DDIM
    print("\n📊 DDIM Scheduler")
    ddim = DDIMScheduler(num_train_steps=1000)
    
    # Sampling timesteps (50 steps au lieu de 1000)
    sampling_steps = ddim.get_sampling_timesteps(num_inference_steps=50)
    print(f"   Training steps: {ddim.num_train_steps}")
    print(f"   Sampling steps: {len(sampling_steps)}")
    print(f"   Timesteps: {sampling_steps[:5].tolist()}...")
    
    # Forward
    x_noisy_ddim = ddim.add_noise(x_clean, timesteps, noise)
    print(f"   Noisy shape: {x_noisy_ddim.shape}")
    
    # Reverse (DDIM step)
    t = 500
    t_next = 450
    x_next, pred_x0 = ddim.step(x_noisy_ddim, t, t_next, predicted_noise, eta=0.0)
    print(f"   Next step shape: {x_next.shape}")
    
    print("\n✅ Schedulers OK!")
    print("\n📝 Notes:")
    print("   - DDPM: 1000 steps, stochastique")
    print("   - DDIM: 50-250 steps, déterministe")
    print("   - Training: utiliser DDPM")
    print("   - Inference: utiliser DDIM (plus rapide)")