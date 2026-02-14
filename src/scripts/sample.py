"""
采样脚本.

从 JSON 配置文件读取采样参数和模型配置, 加载训练好的模型, 采样生成图片并保存.
在 output 目录下创建与配置文件同名的子文件夹.

用法:
    python src/scripts/sample.py configs/zit_sample.json
    python src/scripts/sample.py configs/bat_sample.json
"""

import os
import sys
import json
import argparse
import torch
from torchvision.utils import make_grid, save_image

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import SampleConfig, create_model
from sampler import FlowMatchingSampler


def main():
    parser = argparse.ArgumentParser(description="Sample images from trained model")
    parser.add_argument("config", type=str, help="JSON 配置文件路径")
    args = parser.parse_args()

    # ---- 读取配置 ----
    config_path = args.config
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = SampleConfig(**raw)
    print(f"Config: {config_path} ({config_name})")
    print(f"Model type: {cfg.model.model_type}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 随机种子
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        print(f"Random seed: {cfg.seed}")

    # ---- 输出目录: output_dir / config_name / ... ----
    run_dir = os.path.join(cfg.output_dir, config_name)
    os.makedirs(run_dir, exist_ok=True)

    # ---- 加载模型 ----
    print(f"Loading model from {cfg.model_path}...")
    model = create_model(cfg.model, device)
    checkpoint = torch.load(cfg.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_loss = checkpoint.get("best_test_loss", "?")
    best_ep = checkpoint.get("best_epoch", "?")
    print(f"Model from epoch {epoch}, best_test_loss={best_loss} (epoch {best_ep})")

    # ---- 采样 ----
    sampler = FlowMatchingSampler(num_steps=cfg.sample_steps)
    print(f"Sampling {cfg.num_images} images with {cfg.sample_steps} steps...")

    img_size = cfg.model.img_size
    in_channels = cfg.model.in_channels
    shape = (cfg.num_images, in_channels, img_size, img_size)
    with torch.no_grad():
        images = sampler.sample(model, shape, device)

    # clamp 到 [-1, 1] 然后映射到 [0, 1]
    images = torch.clamp(images, -1, 1)
    images = (images + 1) / 2

    # ---- 保存 ----
    # 保存拼接大图
    output_path = os.path.join(run_dir, cfg.output_filename)
    grid = make_grid(images, nrow=cfg.nrow, padding=2)
    save_image(grid, output_path)
    print(f"Grid image saved to {output_path}")

    # 同时保存单独的图片
    individual_dir = os.path.join(run_dir, os.path.splitext(cfg.output_filename)[0] + "_individual")
    os.makedirs(individual_dir, exist_ok=True)
    for i in range(cfg.num_images):
        save_image(images[i], os.path.join(individual_dir, f"{i:03d}.png"))
    print(f"Individual images saved to {individual_dir}/")

    print("Done!")


if __name__ == "__main__":
    main()
