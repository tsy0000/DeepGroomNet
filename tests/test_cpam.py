"""CPAM 单元测试 + 维度验证 (可开关)。

运行:
    python -m tests.test_cpam
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.modules.cpam import CPAM

# (N, Ci, T, H, W) 官方 TAdaConvNeXtV2-Tiny, t_stride=1, T=32
STAGE_CHANNELS = [96, 192, 384, 768]
T, H, W = 32, 56, 56
NUM_CLASSES = 5


def test_dimensions():
    model = CPAM(stage_channels=STAGE_CHANNELS, num_classes=NUM_CLASSES)
    feats = [
        torch.randn(2, 96, 32, 56, 56),
        torch.randn(2, 192, 32, 28, 28),
        torch.randn(2, 384, 32, 14, 14),
        torch.randn(2, 768, 32, 7, 7),
    ]
    Y, aux = model(feats)
    assert Y.shape == (2, 768, 32, 7, 7), f"Y 维度错误: {Y.shape}"
    assert aux.shape == (2, NUM_CLASSES), f"aux 维度错误: {aux.shape}"
    print(f"[OK] Y={tuple(Y.shape)} aux={tuple(aux.shape)}")


def test_enabled_switch():
    """可开关: temporal_mixer=False 应同样输出正确维度."""
    model = CPAM(stage_channels=STAGE_CHANNELS, num_classes=NUM_CLASSES,
                 temporal_mixer=False)
    feats = [torch.randn(1, c, 32, 56 // (2 ** i), 56 // (2 ** i))
             for i, c in enumerate(STAGE_CHANNELS)]
    Y, aux = model(feats)
    assert Y.shape == (1, 768, 32, 7, 7)
    assert aux.shape == (1, NUM_CLASSES)
    print(f"[OK] no-mixer Y={tuple(Y.shape)} aux={tuple(aux.shape)}")


def test_gradient_flow():
    """梯度应能从 aux_loss 回传到所有 stage 特征."""
    model = CPAM(stage_channels=STAGE_CHANNELS, num_classes=NUM_CLASSES)
    feats = [torch.randn(2, c, 32, 56 // (2 ** i), 56 // (2 ** i), requires_grad=True)
             for i, c in enumerate(STAGE_CHANNELS)]
    _, aux = model(feats)
    loss = aux.sum()
    loss.backward()
    grads = [f.grad is not None and f.grad.abs().sum() > 0 for f in feats]
    assert all(grads), f"部分 stage 特征无梯度: {grads}"
    print(f"[OK] 4 个 stage 特征均有梯度回传: {grads}")


if __name__ == "__main__":
    test_dimensions()
    test_enabled_switch()
    test_gradient_flow()
    print("CPAM 全部测试通过")