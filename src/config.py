"""
配置定义和模型工厂.

包含:
- ZiT / BAT 模型配置 (pydantic BaseModel)
- 训练配置 / 采样配置
- 模型工厂函数
"""

from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field

from models.zit import ZiT
from models.bat import BAT


# ============================================================
# Model Configs
# ============================================================

class ZiTModelConfig(BaseModel):
    """ZiT 模型配置."""
    model_type: Literal["zit"] = "zit"
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_embed_dim: int = 256
    cond_dim: int = 256


class BATModelConfig(BaseModel):
    """BAT 模型配置."""
    model_type: Literal["bat"] = "bat"
    img_size: int = 256
    in_channels: int = 3
    input_dim: int = 128
    latent_dim: int = 512
    latent_len: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_embed_dim: int = 256
    cond_dim: int = 256


ModelConfig = Annotated[
    Union[ZiTModelConfig, BATModelConfig],
    Field(discriminator="model_type"),
]


# ============================================================
# Train / Sample Configs
# ============================================================

class TrainConfig(BaseModel):
    """训练配置."""
    model: Union[ZiTModelConfig, BATModelConfig] = Field(discriminator="model_type")

    # 数据
    train_dir: str = "dataset/train"
    test_dir: str = "dataset/test"

    # 训练超参
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 4

    # 采样
    sample_steps: int = 50

    # 输出
    output_dir: str = "output"

    # 断点续训
    resume: bool = False


class SampleConfig(BaseModel):
    """采样配置."""
    model: Union[ZiTModelConfig, BATModelConfig] = Field(discriminator="model_type")

    # 模型权重路径
    model_path: str = "output/models/best.model"

    # 采样参数
    num_images: int = 16
    nrow: int = 4
    sample_steps: int = 50
    seed: int | None = None

    # 输出
    output_dir: str = "output"
    output_filename: str = "sampled.png"


# ============================================================
# Factory
# ============================================================

def create_model(config: ZiTModelConfig | BATModelConfig, device: str = "cpu"):
    """根据配置创建模型.

    Args:
        config: ZiTModelConfig 或 BATModelConfig 实例.
        device: 设备.

    Returns:
        对应的模型实例, 已移动到指定设备.
    """
    if isinstance(config, ZiTModelConfig):
        model = ZiT(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            time_embed_dim=config.time_embed_dim,
            cond_dim=config.cond_dim,
        )
    elif isinstance(config, BATModelConfig):
        model = BAT(
            img_size=config.img_size,
            in_channels=config.in_channels,
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            latent_len=config.latent_len,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            time_embed_dim=config.time_embed_dim,
            cond_dim=config.cond_dim,
        )
    else:
        raise ValueError(f"Unknown model config type: {type(config)}")

    return model.to(device)
