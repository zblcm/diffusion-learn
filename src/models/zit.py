"""
ZiT (Zigzag Transformer) for image generation with flow matching.
Processes image patches with zigzag ordering.
"""

import torch
import torch.nn as nn

from .shared import (
    RMSNorm,
    SelfAttentionBlock,
    TimestepConditioner,
    init_weights,
)


class PatchEmbed(nn.Module):
    """将图像分割为 patch 并嵌入到向量空间."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
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


class PatchUnembed(nn.Module):
    """将 patch 嵌入还原为图像."""

    def __init__(self, img_size, patch_size, out_channels, embed_dim):
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

    输入: noisy image (B, 3, img_size, img_size) + timestep (B,)
    输出: predicted velocity field (B, 3, img_size, img_size)
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
        time_embed_dim,
        cond_dim,
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
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout, cond_dim)
            for _ in range(depth)
        ])

        # 最终 norm 和输出
        self.final_norm = RMSNorm(embed_dim)
        self.patch_unembed = PatchUnembed(img_size, patch_size, in_channels, embed_dim)

        init_weights(self)

    def forward(self, x, t):
        """
        x: (B, 3, img_size, img_size) noisy image
        t: (B,) timestep in [0, 1]
        returns: (B, 3, img_size, img_size) predicted velocity
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


def create_zit_model(device="cpu"):
    """创建默认配置的 ZiT 模型."""
    model = ZiT(
        img_size=256,
        patch_size=16,
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
