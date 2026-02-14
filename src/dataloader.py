"""
数据加载器, 包含缩放、平移、镜像等数据增强.
从 dataset 文件夹加载 RGB 图片, 输出归一化到 [-1, 1] 的 256x256 tensor.
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CarImageDataset(Dataset):
    """汽车图片数据集."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root_dir, img_size=256, augment=True):
        """
        Args:
            root_dir: 图片文件夹路径
            img_size: 输出图片大小
            augment: 是否启用数据增强
        """
        self.root_dir = root_dir
        self.img_size = img_size

        # 收集所有图片路径
        self.image_paths = []
        for fname in sorted(os.listdir(root_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in self.EXTENSIONS:
                self.image_paths.append(os.path.join(root_dir, fname))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # 构建 transform
        if augment:
            self.transform = transforms.Compose([
                # 缩放: 先 resize 到稍大尺寸, 再随机裁剪
                transforms.Resize(int(img_size * 1.15)),
                transforms.RandomCrop(img_size),
                # 平移: RandomAffine 实现随机平移
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),  # 最大平移 10%
                ),
                # 镜像: 随机水平翻转
                transforms.RandomHorizontalFlip(p=0.5),
                # 转 tensor 并归一化到 [-1, 1]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img


def create_dataloaders(
    data_dir,
    test_dir,
    img_size=256,
    batch_size=32,
    num_workers=4,
):
    """
    创建训练和测试数据加载器.
    
    Args:
        data_dir: 训练集文件夹路径
        test_dir: 测试集文件夹路径
        img_size: 图片大小
        batch_size: 批大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, test_loader
    """

    # 创建带增强的训练集和不带增强的测试集
    train_dataset = CarImageDataset(data_dir, img_size, augment=True)
    test_dataset = CarImageDataset(test_dir, img_size, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
