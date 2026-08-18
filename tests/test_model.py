"""TadaPandaModel 完整模型测试 + 维度验证 (可开关)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.tadapanda_model import TAdaPandaModel


def test_full_t32():
    """T=32, 双模块开启: 主头 + 辅助头维度正确."""
    model = TAdaPandaModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=True)
    x = torch.randn(2, 3, 32, 224, 224)   # (N,C,T,H,W)
    logits, aux = model(x)
    assert logits.shape == (2, 5), f"logits: {logits.shape}"
    assert aux is not None and aux.shape == (2, 5), f"aux: {aux.shape}"
    n = sum(p.numel() for p in model.parameters())
    print(f"[OK] T32 双模块: logits={tuple(logits.shape)} aux={tuple(aux.shape)} 参数量={n/1e6:.1f}M")


def test_cpam_only():
    """仅 CPAM, 无 STSA."""
    model = TAdaPandaModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=False)
    x = torch.randn(1, 3, 32, 224, 224)
    logits, aux = model(x)
    assert logits.shape == (1, 5) and aux.shape == (1, 5)
    assert model.stsa3 is None and model.stsa4 is None
    print(f"[OK] 仅CPAM: logits={tuple(logits.shape)} aux={tuple(aux.shape)}")


def test_stsa_only():
    """仅 STSA, 无 CPAM (aux 应为 None)."""
    model = TAdaPandaModel(num_classes=5, t_stride=1, use_cpam=False, use_stsa=True)
    x = torch.randn(1, 3, 32, 224, 224)
    logits, aux = model(x)
    assert logits.shape == (1, 5)
    assert aux is None
    print(f"[OK] 仅STSA: logits={tuple(logits.shape)} aux={aux}")


def test_t16():
    """T=16 (t_stride=2, 官方默认): 主头输出仍正确."""
    model = TAdaPandaModel(num_classes=5, t_stride=2, use_cpam=True, use_stsa=True)
    x = torch.randn(1, 3, 32, 224, 224)   # 输入仍32帧, stem 下采样到16
    logits, aux = model(x)
    assert logits.shape == (1, 5) and aux.shape == (1, 5)
    print(f"[OK] T16: logits={tuple(logits.shape)} aux={tuple(aux.shape)}")


def test_backward():
    """正反向传播: 双损失回传."""
    model = TAdaPandaModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=True)
    x = torch.randn(2, 3, 32, 224, 224)
    y = torch.zeros(2, 5); y[:, 0] = 1
    logits, aux = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + \
           torch.nn.functional.binary_cross_entropy_with_logits(aux, y)
    loss.backward()
    assert logits.grad is None  # 非叶子
    print(f"[OK] 正反向传播 loss={loss.item():.3f} 正常回传")


if __name__ == "__main__":
    test_full_t32()
    test_cpam_only()
    test_stsa_only()
    test_t16()
    test_backward()
    print("TadaPandaModel 全部测试通过")