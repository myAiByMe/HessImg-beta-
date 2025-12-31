#!/usr/bin/env python3
"""
📝 Text Encoder - Lightweight Transformer
Tokens → Embeddings (77×768)
~120M parameters (version compacte de CLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextEmbedding(nn.Module):
    """Token + Positional embeddings"""
    def __init__(self, vocab_size, embed_dim, max_length):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_length, embed_dim)
        self.max_length = max_length
    
    def forward(self, token_ids):
        """
        token_ids: [batch, seq_len]
        returns: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_emb = self.token_embed(token_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embed(positions)
        
        return token_emb + pos_emb

class TextAttention(nn.Module):
    """Multi-head self-attention pour texte"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, embed_dim]
        mask: [batch, seq_len] (1 pour tokens valides, 0 pour padding)
        """
        batch, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]
        
        # Apply mask si fourni
        if mask is not None:
            mask = mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = attn @ v  # [batch, heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        out = self.proj(out)
        
        return out

class TextFeedForward(nn.Module):
    """FFN avec GELU"""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

class TextTransformerBlock(nn.Module):
    """Bloc transformer complet"""
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = TextAttention(embed_dim, num_heads)
        self.ff = TextFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        # Self-attention avec residual
        x = x + self.attn(self.norm1(x), mask)
        
        # Feedforward avec residual
        x = x + self.ff(self.norm2(x))
        
        return x

class TextEncoder(nn.Module):
    """
    Text Encoder complet
    
    Architecture:
    - Embedding: vocab → 768 dim
    - 6 Transformer layers
    - Output: [batch, 77, 768]
    """
    def __init__(
        self,
        vocab_size=50257,     # GPT-2 vocab size
        embed_dim=768,        # Dimension des embeddings
        max_length=77,        # Longueur max comme CLIP
        num_layers=6,         # Moins de layers que CLIP (12)
        num_heads=12,         # 12 attention heads
        hidden_dim=3072,      # FFN hidden dim
    ):
        super().__init__()
        
        # Embeddings
        self.embedding = TextEmbedding(vocab_size, embed_dim, max_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TextTransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, token_ids, return_mask=False):
        """
        token_ids: [batch, seq_len] (max 77)
        returns: [batch, seq_len, 768]
        """
        # Embeddings
        x = self.embedding(token_ids)
        
        # Mask pour padding (assume que 0 = padding)
        mask = (token_ids != 0).float()
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm
        x = self.final_norm(x)
        
        if return_mask:
            return x, mask
        return x

# ============================================
# TEST
# ============================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Créer encoder
    encoder = TextEncoder(
        vocab_size=50257,
        embed_dim=768,
        max_length=77,
        num_layers=6,
        num_heads=12,
        hidden_dim=3072
    ).to(device)
    
    # Test avec tokenizer
    try:
        from text_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer(max_length=77)
        
        # Test prompts
        prompts = [
            "a beautiful cat sitting on a table",
            "cyberpunk city at night",
        ]
        
        # Tokenize
        token_ids = tokenizer.encode_batch(prompts).to(device)
        
        # Encode
        embeddings = encoder(token_ids)
        
        print("="*60)
        print("📝 TEXT ENCODER TEST")
        print("="*60)
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Expected: [batch={len(prompts)}, seq_len=77, dim=768]")
    except ImportError:
        print("="*60)
        print("📝 TEXT ENCODER TEST (without tokenizer)")
        print("="*60)
        # Test sans tokenizer
        token_ids = torch.randint(0, 50257, (2, 77)).to(device)
        embeddings = encoder(token_ids)
        print(f"Token IDs shape: {token_ids.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
    
    # Parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n📊 Parameters: {num_params/1e6:.1f}M")
    
    # Memory
    if device == "cuda":
        print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n✅ Text Encoder OK!")