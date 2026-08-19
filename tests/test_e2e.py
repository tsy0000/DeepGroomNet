"""端到端测试: 过拟合小数据集 (验证模型能学会记忆训练集).

用合成数据 (有限样本) 训练, 观察训练 loss 显著下降, 证明
TAdaConvModel + 条件损失 + Lightning 全链路可收敛。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train.lightning_module import TAdaConvLightning
from models.tadaconv_model import TAdaConvModel

CFG = {
    "MODEL": {"cpam": True, "stsa": True, "cpam_alpha": 0.9, "cpam_beta": 0.1,
              "drop_path": 0.0, "pretrained": ""},
    "LOSS": {"binary_gamma": 1.0},
    "TRAIN": {"lr": 1e-3, "head_lr_mult": 1.0, "weight_decay": 0.0,
              "epochs": 60, "l2sp_weight": 0.0},
    "OUTPUT": {"log_dir": "outputs/logs", "checkpoint_dir": "outputs/checkpoints"},
}


def make_synthetic(n=6, T=8, size=32, num_classes=5):
    """合成可记忆的小数据集: 每样本固定类别 (可区分)."""
    x = torch.randn(n, 3, T, size, size) * 0.3 + torch.arange(n).float().view(n, 1, 1, 1, 1) * 0.05
    y = torch.arange(n) % num_classes
    return x, y


def run_overfit(epochs=5):
    """训练一小段, 断言 train loss 下降."""
    x, y = make_synthetic(T=8, size=32)   # 小 T/尺寸 加速 CPU 验证
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    m = TAdaConvLightning(CFG, num_classes=5, pos_weight=None)
    m.set_phase("lpft")   # 解冻全部 (小数据快速过拟合验证)

    optimizers, _ = m.configure_optimizers()
    opt = optimizers[0]   # Lightning 返回 [optimizer], [scheduler]
    losses = []
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = m.training_step((xb, yb), 0)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(dl))
        print(f"  ep{ep+1}: loss={losses[-1]:.4f}")

    assert losses[-1] < losses[0], f"loss 应下降: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"[OK] 过拟合验证: loss {losses[0]:.4f} -> {losses[-1]:.4f} 显著下降")


if __name__ == "__main__":
    run_overfit(epochs=8)
    print("端到端过拟合验证通过")