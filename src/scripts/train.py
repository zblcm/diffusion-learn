"""
训练/测试脚本.

功能:
- 从 JSON 配置文件读取训练参数和模型配置
- 通过工厂函数创建 ZiT 或 BAT 模型
- 在 output 目录下创建与配置文件同名的子文件夹, 所有输出放在其中
- 每个 epoch 结束后计算 train/test loss
- 每个 epoch 结束后采样 16 张图片拼成一张保存
- 每 25 个 epoch 保存一次模型
- 每个 epoch 更新 last.model 和 best.model
- 支持断点继续训练
- TensorBoard 可视化

用法:
    python src/scripts/train.py configs/zit_default.json
    python src/scripts/train.py configs/bat_default.json
"""

import os
import sys
import json
import shutil
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import TrainConfig, create_model
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

    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch.to(device)
        optimizer.zero_grad()
        loss = FlowMatchingSampler.compute_loss(model, images, device)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # 实时显示当前 batch 的 loss
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def test_one_epoch(model, test_loader, device):
    """测试一个 epoch, 返回平均 loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(test_loader, desc="Test", leave=False)
    for batch in pbar:
        images = batch.to(device)
        loss = FlowMatchingSampler.compute_loss(model, images, device)
        total_loss += loss.item()
        num_batches += 1
        
        # 实时显示当前 batch 的 loss
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / max(num_batches, 1)


def sample_images(model, sampler, device, img_size, in_channels, num_images=16):
    """采样生成图片, 返回 (num_images, C, H, W) tensor."""
    model.eval()
    shape = (num_images, in_channels, img_size, img_size)
    images = sampler.sample(model, shape, device)
    # clamp 到 [-1, 1] 然后映射到 [0, 1]
    images = torch.clamp(images, -1, 1)
    images = (images + 1) / 2
    return images


def main():
    parser = argparse.ArgumentParser(description="Train model for image generation")
    parser.add_argument("config", type=str, help="JSON 配置文件路径")
    args = parser.parse_args()

    # ---- 读取配置 ----
    config_path = args.config
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = TrainConfig(**raw)
    print(f"Config: {config_path} ({config_name})")
    print(f"Model type: {cfg.model.model_type}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 输出目录: output_dir / config_name / ... ----
    run_dir = os.path.join(cfg.output_dir, config_name)
    images_dir = os.path.join(run_dir, "images")
    errors_dir = os.path.join(run_dir, "errors")
    models_dir = os.path.join(run_dir, "models")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # 拷贝配置文件到输出目录
    dst_config = os.path.join(run_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst_config)
    print(f"Config copied to {dst_config}")

    # ---- 数据加载 ----
    print("Loading dataset...")
    train_loader, test_loader = create_dataloaders(
        cfg.train_dir,
        cfg.test_dir,
        img_size=cfg.model.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # ---- 模型 ----
    model = create_model(cfg.model, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # 采样器
    sampler = FlowMatchingSampler(num_steps=cfg.sample_steps)

    # ---- 断点继续训练 ----
    start_epoch = 0
    best_test_loss = float("inf")
    best_epoch = 0

    last_model_path = os.path.join(models_dir, "last.model")
    if cfg.resume and os.path.exists(last_model_path):
        print(f"Resuming from {last_model_path}...")
        start_epoch, best_test_loss, best_epoch = load_checkpoint(
            last_model_path, model, optimizer, device
        )
        start_epoch += 1  # 从下一个 epoch 开始
        print(f"Resumed at epoch {start_epoch}, best_test_loss={best_test_loss:.6f} (epoch {best_epoch})")

    # TensorBoard
    writer = SummaryWriter(log_dir=tb_dir)

    # ---- 训练循环 ----
    print(f"\nStarting training from epoch {start_epoch} to {cfg.epochs - 1}...")
    print("=" * 60)

    img_size = cfg.model.img_size
    in_channels = cfg.model.in_channels

    for epoch in range(start_epoch, cfg.epochs):
        # ---- 训练 ----
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # ---- 测试 ----
        test_loss = test_one_epoch(model, test_loader, device)

        print(f"Epoch [{epoch:5d}/{cfg.epochs}]  train_loss={train_loss:.6f}  test_loss={test_loss:.6f}", end="")

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
        images = sample_images(model, sampler, device, img_size, in_channels, num_images=16)
        grid = make_grid(images, nrow=4, padding=2)
        img_path = os.path.join(images_dir, f"{epoch:05d}.png")
        save_image(grid, img_path)
        writer.add_image("Samples", grid, epoch)
        print(f" saved to {img_path}")

        # ---- 保存模型 ----
        # 每 25 轮保存一次编号模型
        if epoch % 25 == 0:
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
