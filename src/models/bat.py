"""
BAT (Bottleneck Attention Transformer) for image generation with flow matching.
No patch tokenization. Uses cross-attention to compress a long pixel sequence
into a fixed-length bottleneck, processes it with self-attention, then
cross-attends back to the original sequence length. Perceiver-like architecture.

数据流:
  输入图像 (B,3,H,W)
    → 展平为像素序列 (B, H*W, 3)
    → 线性投影 + 位置编码 → (B, H*W, input_dim)
    → Cross-Attention: latent queries (B, k, latent_dim) attend to input → (B, k, latent_dim)
    → N 层 Self-Attention on bottleneck
    → Cross-Attention: input queries attend to latent → (B, H*W, input_dim)
    → 线性投影 → (B, H*W, 3)
    → reshape → (B, 3, H, W)
"""

import math
import torch
import torch.nn as nn

from .shared import (
    RMSNorm,
    AdaRMSNorm,
    FeedForward,
    MultiHeadCrossAttention,
    SelfAttentionBlock,
    TimestepConditioner,
    init_weights,
)


class SinusoidalPositionalEncoding2D(nn.Module):
    """2D 正弦位置编码, 为每个像素位置提供空间信息."""

    def __init__(self, dim, h, w):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal PE"
        self.dim = dim
        pe = torch.zeros(dim, h, w)
        d_quarter = dim // 4
        # 频率
        div_term = torch.exp(torch.arange(0, d_quarter, dtype=torch.float32) * -(math.log(10000.0) / d_quarter))
        pos_h = torch.arange(0, h, dtype=torch.float32).unsqueeze(1)  # (h, 1)
        pos_w = torch.arange(0, w, dtype=torch.float32).unsqueeze(1)  # (w, 1)
        # sin/cos for height
        pe_h = torch.zeros(dim // 2, h)
        pe_h[0::2, :] = torch.sin(pos_h * div_term).T
        pe_h[1::2, :] = torch.cos(pos_h * div_term).T
        # sin/cos for width
        pe_w = torch.zeros(dim // 2, w)
        pe_w[0::2, :] = torch.sin(pos_w * div_term).T
        pe_w[1::2, :] = torch.cos(pos_w * div_term).T
        # 组合: (dim, h, w)
        pe[:dim // 2, :, :] = pe_h.unsqueeze(2).expand(-1, -1, w)
        pe[dim // 2:, :, :] = pe_w.unsqueeze(1).expand(-1, h, -1)
        # 展平为 (h*w, dim)
        pe = pe.permute(1, 2, 0).reshape(h * w, dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, h*w, dim)

    def forward(self, x):
        # x: (B, h*w, dim)
        return x + self.pe


class CrossAttentionBlock(nn.Module):
    """带自适应归一化的交叉注意力 block."""

    def __init__(self, q_dim, kv_dim, num_heads, mlp_ratio, dropout, cond_dim, skip_q_proj=False):
        super().__init__()
        self.norm_q = AdaRMSNorm(q_dim, cond_dim)
        self.norm_kv = RMSNorm(kv_dim)
        self.cross_attn = MultiHeadCrossAttention(q_dim, kv_dim, num_heads, dropout, skip_q_proj=skip_q_proj)
        self.norm_ffn = AdaRMSNorm(q_dim, cond_dim)
        self.ffn = FeedForward(q_dim, int(q_dim * mlp_ratio), dropout)

    def forward(self, q_seq, kv_seq, cond):
        """
        q_seq: (B, N_q, q_dim)
        kv_seq: (B, N_kv, kv_dim)
        cond: (B, cond_dim)
        returns: (B, N_q, q_dim)
        """
        q_seq = q_seq + self.cross_attn(self.norm_q(q_seq, cond), self.norm_kv(kv_seq))
        q_seq = q_seq + self.ffn(self.norm_ffn(q_seq, cond))
        return q_seq


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        latent_len,
        num_heads,
        mlp_ratio,
        dropout,
        cond_dim
    ):
        super().__init__()
        # encode: Q 来自可学习 latent, 跳过 W_q (latent 直接学习投影后的表示)
        self.encode_cross_attn = CrossAttentionBlock(
            q_dim=latent_dim,
            kv_dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cond_dim=cond_dim,
            skip_q_proj=True,
        )
        self.self_attn = SelfAttentionBlock(
            embed_dim=latent_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cond_dim=cond_dim,
        )
        self.decode_cross_attn = CrossAttentionBlock(
            q_dim=input_dim,
            kv_dim=latent_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cond_dim=cond_dim,
        )

        # 可学习的 bottleneck latent 向量
        self.latent = nn.Parameter(torch.randn(1, latent_len, latent_dim) * 0.02)

        init_weights(self)
        # 重新初始化 latent (init_weights 会覆盖)
        nn.init.trunc_normal_(self.latent, std=0.02)

    def forward(self, x, cond):
        B = x.shape[0]

        # 扩展 latent 到 batch: (1, k, latent_dim) -> (B, k, latent_dim)
        latent = self.latent.expand(B, -1, -1)

        # 编码: latent queries attend to input pixels
        latent = self.encode_cross_attn(latent, x, cond)

        # Self-attention
        latent = self.self_attn(latent, cond)

        # 解码: input queries attend to latent
        x = self.decode_cross_attn(x, latent, cond)

        return x


class BAT(nn.Module):
    """
    BAT: Bottleneck Attention Transformer for image generation with flow matching.

    无 patch tokenization, 通过 cross-attention 瓶颈高效处理长像素序列.

    输入: noisy image (B, 3, img_size, img_size) + timestep (B,)
    输出: predicted velocity field (B, 3, img_size, img_size)
    """

    def __init__(
        self,
        img_size,
        in_channels,
        input_dim,
        latent_dim,
        latent_len,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
        time_embed_dim,
        cond_dim,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.seq_len = img_size * img_size  # 65536 for 256x256

        # 像素投影: 每个像素 (3 channels) -> input_dim
        self.input_proj = nn.Linear(in_channels, input_dim)

        # 2D 正弦位置编码
        self.pos_enc = SinusoidalPositionalEncoding2D(input_dim, img_size, img_size)

        # 时间步条件
        self.time_cond = TimestepConditioner(time_embed_dim, cond_dim)

        # Bottleneck: self-attention blocks on latent sequence
        self.bottleneck_blocks = nn.ModuleList([
            BottleneckBlock(
                input_dim=input_dim,
                latent_dim=latent_dim,
                latent_len=latent_len,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                cond_dim=cond_dim,
            )
            for _ in range(depth)
        ])

        # 输出投影: input_dim -> channels
        self.final_norm = RMSNorm(input_dim)
        self.output_proj = nn.Linear(input_dim, in_channels)

        init_weights(self)

    def forward(self, x, t):
        """
        x: (B, 3, img_size, img_size) noisy image
        t: (B,) timestep in [0, 1]
        returns: (B, 3, img_size, img_size) predicted velocity
        """
        B = x.shape[0]

        # 展平为像素序列: (B, 3, H, W) -> (B, H*W, 3)
        x = x.flatten(2).transpose(1, 2)

        # 像素投影 + 位置编码: (B, H*W, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # 时间条件
        cond = self.time_cond(t)  # (B, cond_dim)

        # Bottleneck self-attention
        for block in self.bottleneck_blocks:
            x = x + block(x, cond)

        # 输出投影: (B, H*W, input_dim) -> (B, H*W, 3)
        x = self.final_norm(x)
        x = self.output_proj(x)

        # Reshape: (B, H*W, 3) -> (B, 3, H, W)
        x = x.transpose(1, 2).reshape(B, self.in_channels, self.img_size, self.img_size)
        return x


def create_bat_model(device="cpu"):
    """创建默认配置的 BAT 模型."""
    model = BAT(
        img_size=256,
        in_channels=3,
        input_dim=128,
        latent_dim=512,
        latent_len=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        time_embed_dim=256,
        cond_dim=256,
    )
    return model.to(device)
