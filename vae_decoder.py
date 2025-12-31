#!/usr/bin/env python3
"""
🎨 VAE Decoder - Compact Version
Latent (4×64×64) → Image (3×512×512)
~25M parameters (miroir de l'encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEResidualBlock(nn.Module):
    """Bloc résiduel pour decoder"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
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
    """Self-attention"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        batch, channels, height, width = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width).transpose(1, 2)
        v = v.reshape(batch, channels, height * width).transpose(1, 2)
        
        scale = channels ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        
        h = attn @ v
        h = h.transpose(1, 2).reshape(batch, channels, height, width)
        h = self.proj(h)
        
        return x + h

class UpsampleBlock(nn.Module):
    """Upsample ×2 avec nearest neighbor + conv"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class VAEDecoder(nn.Module):
    """
    Decoder: 4×64×64 → 3×512×512
    
    Architecture (miroir de l'encoder):
    64×64 → 128×128 → 256×256 → 512×512
    4ch → 512ch → 256ch → 128ch → 3ch
    """
    def __init__(
        self,
        latent_channels=4,
        out_channels=3,
        base_channels=128,
    ):
        super().__init__()
        
        # Input: latent → 512 channels
        self.input_conv = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)
        
        # Middle block (64×64, 512 channels)
        self.middle = nn.Sequential(
            VAEResidualBlock(base_channels * 4, base_channels * 4),
            VAEAttentionBlock(base_channels * 4),
            VAEResidualBlock(base_channels * 4, base_channels * 4),
        )
        
        # Decoder blocks
        # Block 1: 64×64 → 128×128 (512 → 256 channels)
        self.up1 = UpsampleBlock(base_channels * 4)
        self.block1 = nn.Sequential(
            VAEResidualBlock(base_channels * 4, base_channels * 4),
            VAEResidualBlock(base_channels * 4, base_channels * 2),
            VAEAttentionBlock(base_channels * 2),
        )
        
        # Block 2: 128×128 → 256×256 (256 → 128 channels)
        self.up2 = UpsampleBlock(base_channels * 2)
        self.block2 = nn.Sequential(
            VAEResidualBlock(base_channels * 2, base_channels * 2),
            VAEResidualBlock(base_channels * 2, base_channels),
        )
        
        # Block 3: 256×256 → 512×512 (128 channels)
        self.up3 = UpsampleBlock(base_channels)
        self.block3 = nn.Sequential(
            VAEResidualBlock(base_channels, base_channels),
            VAEResidualBlock(base_channels, base_channels),
        )
        
        # Output
        self.output_norm = nn.GroupNorm(32, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, latent):
        """
        latent: [batch, 4, 64, 64]
        returns: [batch, 3, 512, 512] (image)
        """
        # Input
        h = self.input_conv(latent)  # [B, 512, 64, 64]
        
        # Middle
        h = self.middle(h)           # [B, 512, 64, 64]
        
        # Decoder
        h = self.up1(h)              # [B, 512, 128, 128]
        h = self.block1(h)           # [B, 256, 128, 128]
        
        h = self.up2(h)              # [B, 256, 256, 256]
        h = self.block2(h)           # [B, 128, 256, 256]
        
        h = self.up3(h)              # [B, 128, 512, 512]
        h = self.block3(h)           # [B, 128, 512, 512]
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        image = self.output_conv(h)  # [B, 3, 512, 512]
        
        return image

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Créer decoder
    decoder = VAEDecoder(
        latent_channels=4,
        out_channels=3,
        base_channels=128
    ).to(device)
    
    # Test
    latent = torch.randn(2, 4, 64, 64).to(device)
    image = decoder(latent)
    
    print("="*60)
    print("🎨 VAE DECODER TEST")
    print("="*60)
    print(f"Latent shape: {latent.shape}")
    print(f"Image shape:  {image.shape}")
    
    # Parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Parameters: {num_params/1e6:.1f}M")
    
    # Memory
    if device == "cuda":
        print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n✅ Decoder OK!")