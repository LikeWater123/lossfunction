"""Vision Transformer (ViT) for CIFAR image classification.

A compact ViT suitable for 32x32 CIFAR images. Pure PyTorch (no timm).

Default configuration:
    patch_size = 4  -> 8x8 = 64 patches
    embed_dim  = 256
    depth      = 6
    num_heads  = 4
    mlp_ratio  = 2  (hidden_dim = 512)
    dropout    = 0.0

Tokens per image = num_patches (64) + 1 CLS token = 65.
Pre-LayerNorm transformer blocks. Final classification uses the CLS token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """In-place truncated normal init (delegates to ``nn.init.trunc_normal_``).

    Kept as a thin wrapper so call sites read like timm; also serves as a
    fallback shim for older PyTorch versions lacking ``nn.init.trunc_normal_``.
    """
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    # Pure-Python fallback: sample normal then clamp tails.
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)
        tensor.clamp_(min=a, max=b)
        return tensor


class PatchEmbed(nn.Module):
    """Conv-based patch embedding: (B, 3, H, W) -> (B, N, embed_dim).

    Uses a Conv2d with kernel_size == stride == patch_size (no overlap).
    Output tokens are flattened over the spatial dims.
    """

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/ps, W/ps) -> (B, N, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention via scaled_dot_product_attention."""

    def __init__(self, embed_dim: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)
        # SDPA handles scaling internally.
        drop_p = self.attn_drop if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


class MLP(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    def __init__(self, embed_dim: int, hidden_dim: int,
                 drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block.

        x -> LN -> Attn -> (+x) -> LN -> MLP -> (+x)
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_drop=drop,
                                       proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(embed_dim, hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for CIFAR 32x32 inputs."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token (init zeros) and positional embedding (trunc_normal).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=int(embed_dim * mlp_ratio),
                drop=drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        # Spec: CLS token initialized to zeros.
        nn.init.zeros_(self.cls_token)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.forward_features(x)
        return self.head(cls)


def vit_small_cifar(
    num_classes: int = 10,
    patch_size: int = 4,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 4,
    mlp_ratio: float = 2.0,
    drop: float = 0.0,
) -> VisionTransformer:
    """Build a small ViT configured for CIFAR 32x32 inputs."""
    return VisionTransformer(
        img_size=32,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop=drop,
    )
