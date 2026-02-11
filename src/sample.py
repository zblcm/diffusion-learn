"""
采样脚本.
加载训练好的模型, 采样生成图片并保存.
"""

import os
import sys
import argparse
import torch
from torchvision.utils import make_grid, save_image

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from sampler import FlowMatchingSampler


def main():
    parser = argparse.ArgumentParser(description="Sample images from trained ZiT model")
    parser.add_argument("--model_path", type=str, default="output/models/best.model",
                        help="模型路径 (默认使用 best.model)")
    parser.add_argument("--output_path", type=str, default="output/sampled.png",
                        help="输出图片路径")
    parser.add_argument("--num_images", type=int, default=16, help="采样图片数量")
    parser.add_argument("--nrow", type=int, default=4, help="每行图片数量")
    parser.add_argument("--sample_steps", type=int, default=50, help="采样步数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model = create_model(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    best_loss = checkpoint.get("best_test_loss", "?")
    best_ep = checkpoint.get("best_epoch", "?")
    print(f"Model from epoch {epoch}, best_test_loss={best_loss} (epoch {best_ep})")

    # 采样
    sampler = FlowMatchingSampler(num_steps=args.sample_steps)
    print(f"Sampling {args.num_images} images with {args.sample_steps} steps...")

    shape = (args.num_images, 3, 64, 64)
    with torch.no_grad():
        images = sampler.sample(model, shape, device)

    # clamp 到 [-1, 1] 然后映射到 [0, 1]
    images = torch.clamp(images, -1, 1)
    images = (images + 1) / 2

    # 保存
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 保存拼接大图
    grid = make_grid(images, nrow=args.nrow, padding=2)
    save_image(grid, args.output_path)
    print(f"Grid image saved to {args.output_path}")

    # 同时保存单独的图片
    individual_dir = os.path.splitext(args.output_path)[0] + "_individual"
    os.makedirs(individual_dir, exist_ok=True)
    for i in range(args.num_images):
        save_image(images[i], os.path.join(individual_dir, f"{i:03d}.png"))
    print(f"Individual images saved to {individual_dir}/")

    print("Done!")


if __name__ == "__main__":
    main()
