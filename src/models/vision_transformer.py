from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

SUPPORTED_MODELS = [
    "stmae_nano_enc",
    "stmae_nano_dec",
    "stmae_tiny_enc",
    "stmae_tiny_dec",
    "stmae_base_enc",
    "stmae_base_dec",
    "stmae_large_enc",
    "stmae_large_dec"
]

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size: int = 768):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding

        Args:
            x (torch.Tensor): tensor of shape (b, c, h, w)

        Returns:
            torch.Tensor: tensor of shape (b, n, e)
        """
        x = self.projection(x)  # (b, e, h, w)
        h, w = x.shape[2:]
        x = rearrange(x, 'b e h w -> b (h w) e', h=h, w=w)
        # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, n, e) -> (b, n, e')
        return self.net(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.to_out = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Multi-head attention block
        Args:
            x (torch.Tensor): tensor of shape (b, n, e)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tensor of shape (b, n, e), tensor of shape (b, h, n, n)
        """
        # x: (b, n, e)
        b, n, e = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (b, n, e)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)  # (b, h, n, d)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)  # (b, h, n, n)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (b, n, e)
        out = self.to_out(out)
        return out, attn

class PosEmbedding(nn.Module):
    def __init__(self, emb_size, num_tokens):
        super().__init__()
        self.emb_size = emb_size
        self.resolution = num_tokens
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, emb_size))
    
    def forward(self, x: torch.Tensor, apply_indices=None) -> torch.Tensor:
        # x: (b, n, e)
        # apply_indices: (n, )
        if apply_indices is not None:
            x = x + self.pos_embedding[apply_indices]
        else:
            x = x + self.pos_embedding
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio, \
                layer_norm=nn.LayerNorm):
        super().__init__()
        self.in_resolutions = in_resolution
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_patches = (in_resolution // patch_size) ** 2
        
        self.layer_norm = layer_norm(emb_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(emb_size, num_heads),
                FeedForwardBlock(emb_size, int(emb_size * mlp_ratio), emb_size)
            ])
            for _ in range(num_layers)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward pass

        Args:
            x (torch.Tensor): tensor of shape (b, n, e)
        Returns:
            Tuple[torch.Tensor, list]: tensor of shape (b, n, e), list of tensors of shape (b, h, n, n)
        """
        attn_weights_list = []
        for i, (attn, ffn) in enumerate(self.layers):
            # Multi-head attention
            residual = x
            x, attn_weights = attn(self.layer_norm(x))
            attn_weights_list.append(attn_weights)
            x = x + residual
            
            # Feed forward
            residual = x
            x = ffn(self.layer_norm(x))
            x = x + residual
        return x, attn_weights_list


class VisionTransformerDecoder(nn.Module):
    def __init__(self, num_patches, in_channels, emb_size, num_layers, num_heads, mlp_ratio, layer_norm=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pos_embed = PosEmbedding(emb_size, num_patches)
        self.layer_norm = layer_norm(emb_size)
        
        self.embed = nn.Linear(in_channels, emb_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(emb_size, num_heads),
                FeedForwardBlock(emb_size, int(emb_size * mlp_ratio), emb_size)
            ])
            for _ in range(num_layers)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward pass

        Args:
            x (torch.Tensor): tensor of shape (b, n, e)
        Returns:
            Tuple[torch.Tensor, list]: tensor of shape (b, n, e), list of tensors of shape (b, h, n, n)
        """
        attn_weights_list = []
        x = self.embed(x)
        
        for i, (attn, ffn) in enumerate(self.layers):
            # Multi-head attention
            residual = x
            x, attn_weights = attn(self.layer_norm(x))
            attn_weights_list.append(attn_weights)
            x = x + residual
            
            # Feed forward
            residual = x
            x = ffn(self.layer_norm(x))
            x = x + residual
        return x, attn_weights_list

def stmae_nano_enc(in_resolution, in_channels, patch_size, emb_size=120, mlp_ratio=4, num_layers=6, num_heads=12) -> VisionTransformerEncoder:
    return VisionTransformerEncoder(in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_nano_dec(num_patches, in_channels, emb_size=128, mlp_ratio=4, num_layers=4, num_heads=16) -> VisionTransformerDecoder:
    return VisionTransformerDecoder(num_patches, in_channels, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_tiny_enc(in_resolution, in_channels, patch_size, emb_size=240, mlp_ratio=4, num_layers=12, num_heads=12) -> VisionTransformerEncoder:
    return VisionTransformerEncoder(in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_tiny_dec(num_patches, in_channels, emb_size=256, mlp_ratio=4, num_layers=8, num_heads=16) -> VisionTransformerDecoder:
    return VisionTransformerDecoder(num_patches, in_channels, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_base_enc(in_resolution, in_channels, patch_size, emb_size=768, mlp_ratio=4, num_layers=12, num_heads=12) -> VisionTransformerEncoder:
    return VisionTransformerEncoder(in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_base_dec(num_patches, in_channels, emb_size=512, mlp_ratio=4, num_layers=8, num_heads=16) -> VisionTransformerDecoder:
    return VisionTransformerDecoder(num_patches, in_channels, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_large_enc(in_resolution, in_channels, patch_size, emb_size=1024, mlp_ratio=4, num_layers=24, num_heads=16) -> VisionTransformerEncoder:
    return VisionTransformerEncoder(in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio)

def stmae_large_dec(num_patches, in_channels, emb_size=512, mlp_ratio=4, num_layers=8, num_heads=16) -> VisionTransformerDecoder:
    return VisionTransformerDecoder(num_patches, in_channels, emb_size, num_layers, num_heads, mlp_ratio)

if __name__ == '__main__':
    pass