"""
训练/测试脚本.

功能:
- 训练 ZiT 模型进行 flow matching
- 每个 epoch 结束后计算 train/test loss
- 每个 epoch 结束后采样 16 张图片拼成一张保存
- 每 5 个 epoch 保存一次模型
- 每个 epoch 更新 last.model 和 best.model
- 支持断点继续训练
- TensorBoard 可视化
"""

import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model
from sampler import FlowMatchingSampler
from dataloader import create_dataloaders


def save_checkpoint(path, model, optimizer, epoch, best_test_loss, best_epoch):
    """保存模型检查点."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_test_loss": best_test_loss,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """加载模型检查点."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    best_test_loss = checkpoint.get("best_test_loss", float("inf"))
    best_epoch = checkpoint.get("best_epoch", 0)
    return epoch, best_test_loss, best_epoch


def train_one_epoch(model, train_loader, optimizer, device):
    """训练一个 epoch, 返回平均 loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch.to(device)
        optimizer.zero_grad()
        loss = FlowMatchingSampler.compute_loss(model, images, device)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def test_one_epoch(model, test_loader, device):
    """测试一个 epoch, 返回平均 loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in test_loader:
        images = batch.to(device)
        loss = FlowMatchingSampler.compute_loss(model, images, device)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def sample_images(model, sampler, device, num_images=16):
    """采样生成图片, 返回 (num_images, 3, 64, 64) tensor."""
    model.eval()
    shape = (num_images, 3, 64, 64)
    images = sampler.sample(model, shape, device)
    # clamp 到 [-1, 1] 然后映射到 [0, 1]
    images = torch.clamp(images, -1, 1)
    images = (images + 1) / 2
    return images


def main():
    parser = argparse.ArgumentParser(description="Train ZiT for car image generation")
    parser.add_argument("--data_dir", type=str, default="dataset/train", help="数据集路径")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出路径")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--sample_steps", type=int, default=50, help="采样步数")
    parser.add_argument("--resume", action="store_true", help="从 last.model 继续训练")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 输出目录
    images_dir = os.path.join(args.output_dir, "images")
    errors_dir = os.path.join(args.output_dir, "errors")
    models_dir = os.path.join(args.output_dir, "models")
    tb_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # 数据加载
    print("Loading dataset...")
    train_loader, test_loader = create_dataloaders(
        args.data_dir,
        args.test_dir,
        img_size=64,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 模型
    model = create_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 采样器
    sampler = FlowMatchingSampler(num_steps=args.sample_steps)

    # 断点继续训练
    start_epoch = 0
    best_test_loss = float("inf")
    best_epoch = 0

    last_model_path = os.path.join(models_dir, "last.model")
    if args.resume and os.path.exists(last_model_path):
        print(f"Resuming from {last_model_path}...")
        start_epoch, best_test_loss, best_epoch = load_checkpoint(
            last_model_path, model, optimizer, device
        )
        start_epoch += 1  # 从下一个 epoch 开始
        print(f"Resumed at epoch {start_epoch}, best_test_loss={best_test_loss:.6f} (epoch {best_epoch})")

    # TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)

    # 训练循环
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs - 1}...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # ---- 训练 ----
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # ---- 测试 ----
        test_loss = test_one_epoch(model, test_loader, device)

        print(f"Epoch [{epoch:5d}/{args.epochs}]  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}", end="")

        # ---- 更新 best ----
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            best_epoch = epoch
            print(f"  ★ new best!", end="")
        print()

        # ---- 保存 errors ----
        error_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "best_test_loss": best_test_loss,
            "best_epoch": best_epoch,
        }
        error_path = os.path.join(errors_dir, f"{epoch:05d}.json")
        with open(error_path, "w") as f:
            json.dump(error_data, f, indent=2)

        # ---- TensorBoard ----
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Loss/best_test", best_test_loss, epoch)

        # ---- 采样图片 ----
        print(f"  Sampling 16 images...", end="", flush=True)
        images = sample_images(model, sampler, device, num_images=16)
        grid = make_grid(images, nrow=4, padding=2)
        img_path = os.path.join(images_dir, f"{epoch:05d}.png")
        save_image(grid, img_path)
        writer.add_image("Samples", grid, epoch)
        print(f" saved to {img_path}")

        # ---- 保存模型 ----
        # 每 5 轮保存一次编号模型
        if epoch % 5 == 0:
            numbered_path = os.path.join(models_dir, f"{epoch:05d}.model")
            save_checkpoint(numbered_path, model, optimizer, epoch, best_test_loss, best_epoch)

        # 每轮更新 last.model
        save_checkpoint(last_model_path, model, optimizer, epoch, best_test_loss, best_epoch)

        # 更新 best.model
        if is_best:
            best_model_path = os.path.join(models_dir, "best.model")
            save_checkpoint(best_model_path, model, optimizer, epoch, best_test_loss, best_epoch)

        print()

    writer.close()
    print("=" * 60)
    print(f"Training complete! Best test loss: {best_test_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
