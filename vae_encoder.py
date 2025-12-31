#!/usr/bin/env python3
"""
🎨 VAE Encoder - Compact Version
Image (3×512×512) → Latent (4×64×64)
~25M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEResidualBlock(nn.Module):
    """Bloc résiduel optimisé pour VAE"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        # Skip connection si channels différents
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual

class VAEAttentionBlock(nn.Module):
    """Self-attention pour capturer les dépendances globales"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape pour attention
        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width).transpose(1, 2)
        v = v.reshape(batch, channels, height * width).transpose(1, 2)
        
        # Attention
        scale = channels ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        
        # Apply
        h = attn @ v
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        h = self.proj(h)
        
        return x + h

class DownsampleBlock(nn.Module):
    """Downsample par 2 avec conv stride=2"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class VAEEncoder(nn.Module):
    """
    Encoder: 3×512×512 → 4×64×64
    
    Architecture:
    512×512 → 256×256 → 128×128 → 64×64
    3ch → 128ch → 256ch → 512ch → 4ch (mean/logvar = 8ch)
    """
    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        base_channels=128,
    ):
        super().__init__()
        
        # Input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder blocks
        # Block 1: 512×512 → 256×256 (128 channels)
        self.block1 = nn.Sequential(
            VAEResidualBlock(base_channels, base_channels),
            VAEResidualBlock(base_channels, base_channels),
        )
        self.down1 = DownsampleBlock(base_channels)
        
        # Block 2: 256×256 → 128×128 (256 channels)
        self.block2 = nn.Sequential(
            VAEResidualBlock(base_channels, base_channels * 2),
            VAEResidualBlock(base_channels * 2, base_channels * 2),
        )
        self.down2 = DownsampleBlock(base_channels * 2)
        
        # Block 3: 128×128 → 64×64 (512 channels)
        self.block3 = nn.Sequential(
            VAEResidualBlock(base_channels * 2, base_channels * 4),
            VAEResidualBlock(base_channels * 4, base_channels * 4),
            VAEAttentionBlock(base_channels * 4),  # Attention au niveau le plus bas
        )
        self.down3 = DownsampleBlock(base_channels * 4)
        
        # Middle block (64×64, 512 channels)
        self.middle = nn.Sequential(
            VAEResidualBlock(base_channels * 4, base_channels * 4),
            VAEAttentionBlock(base_channels * 4),
            VAEResidualBlock(base_channels * 4, base_channels * 4),
        )
        
        # Output: mean et logvar pour reparametrization trick
        self.output_norm = nn.GroupNorm(32, base_channels * 4)
        self.output_conv = nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1)
    
    def forward(self, x):
        """
        x: [batch, 3, 512, 512]
        returns: [batch, 4, 64, 64] (latent)
        """
        # Input
        h = self.input_conv(x)  # [B, 128, 512, 512]
        
        # Encoder
        h = self.block1(h)       # [B, 128, 512, 512]
        h = self.down1(h)        # [B, 128, 256, 256]
        
        h = self.block2(h)       # [B, 256, 256, 256]
        h = self.down2(h)        # [B, 256, 128, 128]
        
        h = self.block3(h)       # [B, 512, 128, 128]
        h = self.down3(h)        # [B, 512, 64, 64]
        
        # Middle
        h = self.middle(h)       # [B, 512, 64, 64]
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)  # [B, 8, 64, 64]
        
        # Split mean et logvar
        mean, logvar = h.chunk(2, dim=1)  # [B, 4, 64, 64] each
        
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        
        return latent, mean, logvar

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Créer encoder
    encoder = VAEEncoder(
        in_channels=3,
        latent_channels=4,
        base_channels=128
    ).to(device)
    
    # Test
    x = torch.randn(2, 3, 512, 512).to(device)
    latent, mean, logvar = encoder(x)
    
    print("="*60)
    print("🎨 VAE ENCODER TEST")
    print("="*60)
    print(f"Input shape:  {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Mean shape:   {mean.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n📊 Parameters: {num_params/1e6:.1f}M")
    
    # Memory
    if device == "cuda":
        print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n✅ Encoder OK!")