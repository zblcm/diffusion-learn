"""
Flow Matching 采样器.

Flow Matching 的核心思想:
- 定义一个从噪声分布 (t=0) 到数据分布 (t=1) 的概率流 ODE
- 训练模型预测速度场 v(x_t, t)
- 采样时从噪声出发, 沿速度场积分到 t=1 得到生成样本

插值公式: x_t = (1 - t) * noise + t * data
速度场目标: v = data - noise
"""

import torch


class FlowMatchingSampler:
    """Flow Matching 采样器, 使用 Euler 方法求解 ODE."""

    def __init__(self, num_steps=50):
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(self, model, shape, device="cpu"):
        """
        从纯噪声采样生成图像.
        
        Args:
            model: 预测速度场的模型, 输入 (x_t, t) 输出 v
            shape: 生成图像的形状, 如 (B, 3, 256, 256)
            device: 设备
            
        Returns:
            生成的图像 tensor, 值域约 [-1, 1]
        """
        model.eval()

        # 从标准正态分布采样初始噪声 (t=0)
        x = torch.randn(shape, device=device)

        # 时间步从 0 到 1
        dt = 1.0 / self.num_steps
        timesteps = torch.linspace(0, 1 - dt, self.num_steps, device=device)

        for t_val in timesteps:
            t = torch.full((shape[0],), t_val, device=device)
            # 预测速度场
            v = model(x, t)
            # Euler 步进: x_{t+dt} = x_t + v * dt
            x = x + v * dt

        return x

    @staticmethod
    def compute_loss(model, x1, device="cpu"):
        """
        计算 flow matching 训练损失.
        
        Args:
            model: 预测速度场的模型
            x1: 真实数据 (B, 3, 256, 256), 值域 [-1, 1]
            device: 设备
            
        Returns:
            MSE loss
        """
        B = x1.shape[0]

        # 采样随机时间步 t ~ U(0, 1)
        t = torch.rand(B, device=device)

        # 采样噪声 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 线性插值: x_t = (1 - t) * x0 + t * x1
        t_expand = t[:, None, None, None]  # (B, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # 目标速度: v = x1 - x0
        v_target = x1 - x0

        # 模型预测速度
        v_pred = model(x_t, t)

        # MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss
