# ZiT 汽车图片生成

基于 **ZiT (Zigzag Transformer)** 架构和 **Flow Matching** 的 256×256 RGB 汽车图片生成项目。

## 项目结构

```
test-image/
├── dataset/              # 放入汽车图片 (jpg/png/bmp/webp 等)
├── output/
│   ├── images/           # 每个 epoch 采样的 16 张拼接图
│   ├── errors/           # 每个 epoch 的 train/test loss (JSON)
│   ├── models/           # 模型检查点
│   └── tensorboard/      # TensorBoard 日志
├── src/
│   ├── model.py          # ZiT 模型 (所有模块从零实现)
│   ├── sampler.py        # Flow Matching 采样器
│   ├── dataloader.py     # 数据加载 (含缩放/平移/镜像增强)
│   ├── train.py          # 训练/测试脚本
│   └── sample.py         # 采样脚本
└── README.md
```

## 依赖

```bash
pip install torch torchvision tensorboard pillow
```

## 使用方法

### 1. 准备数据

将汽车图片放入 `dataset/` 文件夹。支持 jpg, jpeg, png, bmp, webp, tiff 格式。

### 2. 训练

```bash
# 从头开始训练
python src/train.py --data_dir dataset --output_dir output --epochs 500 --batch_size 32 --lr 1e-4

# 断点继续训练
python src/train.py --data_dir dataset --output_dir output --epochs 500 --resume
```

**参数说明:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `dataset` | 数据集路径 |
| `--output_dir` | `output` | 输出路径 |
| `--epochs` | `500` | 训练轮数 |
| `--batch_size` | `32` | 批大小 |
| `--lr` | `1e-4` | 学习率 |
| `--num_workers` | `4` | 数据加载线程数 |
| `--sample_steps` | `50` | 采样步数 |
| `--resume` | `False` | 从 last.model 继续训练 |

### 3. 查看训练过程

```bash
tensorboard --logdir output/tensorboard
```

### 4. 采样生成图片

```bash
# 使用 best 模型采样
python src/sample.py --model_path output/models/best.model --output_path output/sampled.png

# 指定参数
python src/sample.py --model_path output/models/best.model --num_images 16 --sample_steps 100 --seed 42
```

## 模型输出说明

### images/
每个 epoch 结束后采样 16 张图片, 拼成 4×4 网格, 保存为 `%05d.png`。

### errors/
每个 epoch 保存一个 JSON 文件, 包含:
- `train_loss`: 训练损失
- `test_loss`: 测试损失
- `best_test_loss`: 历史最小测试损失
- `best_epoch`: 最小损失对应的 epoch

### models/
- `%05d.model`: 每 5 个 epoch 保存一次
- `last.model`: 最后一个 epoch 的模型 (用于断点继续)
- `best.model`: 测试损失最小的模型

每个 `.model` 文件包含:
- 模型权重 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 当前 epoch
- 历史最小 test loss 及对应 epoch

## 架构说明

### ZiT (Zigzag Transformer)
- **PatchEmbed**: 将 256×256 图像分割为 4×4 的 patch, 得到 16×16=256 个 token
- **ZigzagReorder**: 对 patch 序列进行 zigzag 排列, 使空间相邻的 patch 在序列中也相邻
- **MultiHeadSelfAttention**: 8 头自注意力, 从零实现 Q/K/V 投影和注意力计算
- **FeedForward**: GELU 激活的两层 MLP
- **AdaRMSNorm**: 自适应 RMSNorm, 用时间步条件调制 scale/shift
- **TimestepConditioner**: 正弦位置编码 + MLP 将时间步映射为条件向量
- **PatchUnembed**: 将 token 序列还原为图像

### Flow Matching
- 插值: `x_t = (1-t) * noise + t * data`
- 速度目标: `v = data - noise`
- 采样: 从噪声出发, 用 Euler 方法沿预测速度场积分到 t=1
