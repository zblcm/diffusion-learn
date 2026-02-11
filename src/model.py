"""
ZiT (Zigzag Transformer) for image generation.
A simplified Vision Transformer that processes image patches with zigzag ordering.
All modules are implemented from scratch without using pre-built transformer layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """将图像分割为 patch 并嵌入到向量空间."""

    def __init__(self, img_size=64, patch_size=4, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        # 用卷积实现 patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        return x


class ZigzagReorder(nn.Module):
    """对 patch 序列进行 zigzag 排列, 使空间上相邻的 patch 在序列中也相邻."""

    def __init__(self, h, w):
        super().__init__()
        # 预计算 zigzag 顺序的索引
        order = self._zigzag_order(h, w)
        self.register_buffer("order", torch.tensor(order, dtype=torch.long))
        # 计算逆序索引用于还原
        inv_order = [0] * len(order)
        for i, o in enumerate(order):
            inv_order[o] = i
        self.register_buffer("inv_order", torch.tensor(inv_order, dtype=torch.long))

    @staticmethod
    def _zigzag_order(h, w):
        """生成 h x w 网格的 zigzag 遍历顺序."""
        order = []
        for i in range(h):
            if i % 2 == 0:
                for j in range(w):
                    order.append(i * w + j)
            else:
                for j in range(w - 1, -1, -1):
                    order.append(i * w + j)
        return order

    def forward(self, x):
        # x: (B, N, C) -> zigzag reordered (B, N, C)
        return x[:, self.order, :]

    def inverse(self, x):
        # zigzag reordered (B, N, C) -> original order (B, N, C)
        return x[:, self.inv_order, :]


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力, 从零实现."""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        # 输出投影
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        # 计算 Q, K, V 并 reshape 为多头
        q = self.W_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力分数: (B, heads, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    """前馈网络 (MLP), 从零实现."""

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    """RMSNorm, 从零实现."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AdaRMSNorm(nn.Module):
    """自适应 RMSNorm, 用时间步条件调制 scale 和 shift."""

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        # cond: (B, cond_dim)
        scale_shift = self.linear(cond)  # (B, dim*2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # 各 (B, dim)
        scale = scale.unsqueeze(1)  # (B, 1, dim)
        shift = shift.unsqueeze(1)  # (B, 1, dim)
        x = self.norm(x)
        return x * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Transformer block with adaptive normalization for conditioning."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=256):
        super().__init__()
        self.norm1 = AdaRMSNorm(embed_dim, cond_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = AdaRMSNorm(embed_dim, cond_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.ffn(self.norm2(x, cond))
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """正弦时间步嵌入, 从零实现."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) 值域 [0, 1]
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepConditioner(nn.Module):
    """将时间步嵌入映射到条件向量."""

    def __init__(self, time_embed_dim, cond_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class PatchUnembed(nn.Module):
    """将 patch 嵌入还原为图像."""

    def __init__(self, img_size=64, patch_size=4, out_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, N, embed_dim)
        B, N, _ = x.shape
        x = self.proj(x)  # (B, N, P*P*C)
        P = self.patch_size
        C = self.out_channels
        x = x.reshape(B, self.num_patches_h, self.num_patches_w, P, P, C)
        # (B, H_patches, W_patches, P, P, C) -> (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.img_size, self.img_size)
        return x


class ZiT(nn.Module):
    """
    ZiT: Zigzag Transformer for image generation with flow matching.
    
    输入: noisy image (B, 3, 64, 64) + timestep (B,)
    输出: predicted velocity field (B, 3, 64, 64)
    """

    def __init__(
        self,
        img_size=64,
        patch_size=4,
        in_channels=3,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        time_embed_dim=256,
        cond_dim=256,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        h = self.patch_embed.num_patches_h
        w = self.patch_embed.num_patches_w

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Zigzag 重排
        self.zigzag = ZigzagReorder(h, w)

        # 时间步条件
        self.time_cond = TimestepConditioner(time_embed_dim, cond_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, cond_dim)
            for _ in range(depth)
        ])

        # 最终 norm 和输出
        self.final_norm = RMSNorm(embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, in_channels, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, t):
        """
        x: (B, 3, 64, 64) noisy image
        t: (B,) timestep in [0, 1]
        returns: (B, 3, 64, 64) predicted velocity
        """
        # Patch embedding + positional encoding
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed

        # Zigzag reorder
        x = self.zigzag(x)

        # Time conditioning
        cond = self.time_cond(t)  # (B, cond_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Inverse zigzag to restore spatial order
        x = self.zigzag.inverse(x)

        # Final norm + unembed
        x = self.final_norm(x)
        x = self.patch_unembed(x)
        return x


def create_model(device="cpu"):
    """创建默认配置的 ZiT 模型."""
    model = ZiT(
        img_size=64,
        patch_size=4,
        in_channels=3,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
        cond_dim=256,
    )
    return model.to(device)
