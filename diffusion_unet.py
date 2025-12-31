#!/usr/bin/env python3
"""
🎨 Diffusion UNet - Optimized Version
Prédit le bruit dans l'espace latent
~300M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================
# TIMESTEP EMBEDDING
# ============================================
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        """
        timesteps: [batch_size] tensor of ints (0 to num_steps)
        returns: [batch_size, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim)
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

# ============================================
# RESNET BLOCK
# ============================================
class ResNetBlock(nn.Module):
    """ResNet block avec time conditioning"""
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
    
    def forward(self, x, time_emb):
        """
        x: [batch, channels, h, w]
        time_emb: [batch, time_dim]
        """
        residual = self.skip(x)
        
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual

# ============================================
# CROSS ATTENTION
# ============================================
class CrossAttention(nn.Module):
    """Cross-attention entre image et texte"""
    def __init__(self, dim, context_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x, context):
        """
        x: [batch, hw, dim] (image features flattened)
        context: [batch, seq_len, context_dim] (text embeddings)
        """
        batch, seq_len, _ = x.shape
        
        # Q from image, K,V from text
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape pour multi-head
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        out = self.to_out(out)
        
        return out

# ============================================
# SPATIAL TRANSFORMER
# ============================================
class SpatialTransformer(nn.Module):
    """Combine self-attention + cross-attention"""
    def __init__(self, channels, context_dim, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        
        self.cross_attn = CrossAttention(channels, context_dim, num_heads)
        self.norm_cross = nn.LayerNorm(channels)
        
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.norm_ff = nn.LayerNorm(channels)
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x, context):
        """
        x: [batch, channels, h, w]
        context: [batch, seq_len, context_dim]
        """
        batch, channels, h, w = x.shape
        
        # Save residual
        residual = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Flatten spatial dims
        x = x.reshape(batch, channels, h * w).transpose(1, 2)  # [batch, hw, channels]
        
        # Cross-attention
        x = x + self.cross_attn(self.norm_cross(x), context)
        
        # Feedforward
        x = x + self.ff(self.norm_ff(x))
        
        # Reshape back
        x = x.transpose(1, 2).reshape(batch, channels, h, w)
        x = self.proj_out(x)
        
        return x + residual

# ============================================
# UNET
# ============================================
class DiffusionUNet(nn.Module):
    """
    UNet pour diffusion dans l'espace latent
    
    Input: [batch, 4, 64, 64] latent + timestep + text
    Output: [batch, 4, 64, 64] predicted noise
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        num_heads=8,
        context_dim=768,
    ):
        super().__init__()
        
        # Time embedding
        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder
        # 64x64
        self.enc_block1_res = ResNetBlock(model_channels, model_channels, time_dim)
        self.enc_block1_attn = SpatialTransformer(model_channels, context_dim, num_heads)
        self.down1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # 32x32
        self.enc_block2_res = ResNetBlock(model_channels, model_channels * 2, time_dim)
        self.enc_block2_attn = SpatialTransformer(model_channels * 2, context_dim, num_heads)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # 16x16
        self.enc_block3_res = ResNetBlock(model_channels * 2, model_channels * 4, time_dim)
        self.enc_block3_attn = SpatialTransformer(model_channels * 4, context_dim, num_heads)
        
        # Middle (16x16)
        self.mid_res1 = ResNetBlock(model_channels * 4, model_channels * 4, time_dim)
        self.mid_attn = SpatialTransformer(model_channels * 4, context_dim, num_heads)
        self.mid_res2 = ResNetBlock(model_channels * 4, model_channels * 4, time_dim)
        
        # Decoder
        # 16x16
        self.dec_block3_res = ResNetBlock(model_channels * 8, model_channels * 4, time_dim)
        self.dec_block3_attn = SpatialTransformer(model_channels * 4, context_dim, num_heads)
        self.up3 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        
        # 32x32
        self.dec_block2_res = ResNetBlock(model_channels * 6, model_channels * 2, time_dim)
        self.dec_block2_attn = SpatialTransformer(model_channels * 2, context_dim, num_heads)
        self.up2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        
        # 64x64
        self.dec_block1_res = ResNetBlock(model_channels * 3, model_channels, time_dim)
        self.dec_block1_attn = SpatialTransformer(model_channels, context_dim, num_heads)
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps, context):
        """
        x: [batch, 4, 64, 64] noisy latent
        timesteps: [batch] timestep values
        context: [batch, 77, 768] text embeddings
        
        returns: [batch, 4, 64, 64] predicted noise
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder with skip connections
        # Block 1: 64x64
        h = self.enc_block1_res(h, time_emb)
        h = self.enc_block1_attn(h, context)
        skip1 = h
        h = self.down1(h)  # 32x32
        
        # Block 2: 32x32
        h = self.enc_block2_res(h, time_emb)
        h = self.enc_block2_attn(h, context)
        skip2 = h
        h = self.down2(h)  # 16x16
        
        # Block 3: 16x16
        h = self.enc_block3_res(h, time_emb)
        h = self.enc_block3_attn(h, context)
        skip3 = h
        
        # Middle
        h = self.mid_res1(h, time_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, time_emb)
        
        # Decoder
        # Block 3: 16x16
        h = torch.cat([h, skip3], dim=1)
        h = self.dec_block3_res(h, time_emb)
        h = self.dec_block3_attn(h, context)
        h = self.up3(h)  # 32x32
        
        # Block 2: 32x32
        h = torch.cat([h, skip2], dim=1)
        h = self.dec_block2_res(h, time_emb)
        h = self.dec_block2_attn(h, context)
        h = self.up2(h)  # 64x64
        
        # Block 1: 64x64
        h = torch.cat([h, skip1], dim=1)
        h = self.dec_block1_res(h, time_emb)
        h = self.dec_block1_attn(h, context)
        
        # Output
        h = self.output_conv(h)
        
        return h

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create UNet
    unet = DiffusionUNet(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        num_heads=8,
        context_dim=768
    ).to(device)
    
    # Test inputs
    batch_size = 2
    latent = torch.randn(batch_size, 4, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    context = torch.randn(batch_size, 77, 768).to(device)
    
    # Forward
    noise_pred = unet(latent, timesteps, context)
    
    print("="*60)
    print("🎨 DIFFUSION UNET TEST")
    print("="*60)
    print(f"Latent shape:    {latent.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Context shape:   {context.shape}")
    print(f"Output shape:    {noise_pred.shape}")
    
    # Parameters
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"\n📊 Parameters: {num_params/1e6:.1f}M")
    
    # Memory
    if device == "cuda":
        print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n✅ UNet OK!")