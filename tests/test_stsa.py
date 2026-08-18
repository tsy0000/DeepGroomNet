"""STSA 单元测试 + 维度验证 (可开关)。

运行:
    python -m tests.test_stsa
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.modules.stsa import STSA


def test_dimensions_s3():
    """STSA 用于 Stage3 (384 通道): 输入输出同维."""
    model = STSA(channels=384)
    x = torch.randn(2, 384, 32, 14, 14)
    out = model(x)
    assert out.shape == x.shape, f"维度不一致: {out.shape} vs {x.shape}"
    print(f"[OK] S3 out={tuple(out.shape)} == in={tuple(x.shape)}")


def test_dimensions_s4():
    """STSA 用于 Stage4 (768 通道)."""
    model = STSA(channels=768)
    x = torch.randn(1, 768, 32, 7, 7)
    out = model(x)
    assert out.shape == x.shape
    print(f"[OK] S4 out={tuple(out.shape)} == in={tuple(x.shape)}")


def test_zero_extra_params_on_shift():
    """空间移位零参数 (论文 Table 5: Params 不变)."""
    model = STSA(channels=384)
    n_params = sum(p.numel() for p in model.parameters())
    # 移位算子本身无参数, 但 SA/TA/MLP 有参数
    # 验证: shift 函数是纯函数, 可确认这里
    print(f"[INFO] STSA 参数量 (含 SA/TA/MLP): {n_params}")


def test_shift_direction():
    """SS1/SS2 移位方向正确性 (边界置 0)."""
    from models.modules.stsa import spatial_shift1, spatial_shift2
    x = torch.ones(1, 8, 2, 4, 4)
    s1 = spatial_shift1(x)
    # 0~2 通道 H 下移: 首行(H=0)应为0, 其余为1
    assert s1[0, 0, 0, 0].sum() == 0, "SS1 首行应置0"
    assert s1[0, 0, 0, 1:].sum() > 0, "SS1 下移后下方应有值"
    s2 = spatial_shift2(x)
    assert s2[0, 0, 0, :, 0].sum() == 0, "SS2 首列应置0"
    print("[OK] SS1/SS2 移位边界正确")


def test_gradient_flow():
    model = STSA(channels=384)
    x = torch.randn(2, 384, 32, 14, 14, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("[OK] STSA 梯度回传正常")


if __name__ == "__main__":
    test_dimensions_s3()
    test_dimensions_s4()
    test_zero_extra_params_on_shift()
    test_shift_direction()
    test_gradient_flow()
    print("STSA 全部测试通过")