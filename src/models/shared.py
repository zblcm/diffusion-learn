"""
Shared modules for image generation models.
All modules are implemented from scratch without using pre-built transformer layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 使用 Flash Attention (不会实例化完整 attention matrix, 显存 O(1))
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.W_o(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力, 从零实现. Q 来自一个序列, K/V 来自另一个序列.

    Args:
        skip_q_proj: 如果为 True, 跳过 Q 投影矩阵. 适用于 Q 来自可学习参数的场景,
            因为可学习参数可以直接学习投影后的表示, W_q 是冗余的.
    """

    def __init__(self, q_dim, kv_dim, num_heads, dropout=0.0, skip_q_proj=False):
        super().__init__()
        assert q_dim % num_heads == 0, "q_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.skip_q_proj = skip_q_proj

        if not skip_q_proj:
            self.W_q = nn.Linear(q_dim, q_dim)
        self.W_k = nn.Linear(kv_dim, q_dim)
        self.W_v = nn.Linear(kv_dim, q_dim)
        self.W_o = nn.Linear(q_dim, q_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q_seq, kv_seq):
        """
        q_seq: (B, N_q, q_dim) - query 序列
        kv_seq: (B, N_kv, kv_dim) - key/value 序列
        returns: (B, N_q, q_dim)
        """
        B, N_q, _ = q_seq.shape
        N_kv = kv_seq.shape[1]

        if self.skip_q_proj:
            q = q_seq.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            q = self.W_q(q_seq).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(kv_seq).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(kv_seq).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 使用 Flash Attention (不会实例化完整 attention matrix, 显存 O(1))
        dropout_p = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.permute(0, 2, 1, 3).reshape(B, N_q, -1)
        out = self.W_o(out)
        return out


class SelfAttentionBlock(nn.Module):
    """带自适应归一化的自注意力 Transformer block."""

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


def init_weights(module):
    """通用的权重初始化."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
