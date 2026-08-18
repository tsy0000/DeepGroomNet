"""条件自适应损失 (式8-10) 单元测试."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from losses.auxiliary import ConditionalLoss, FocalBCE, AuxFocalLoss


def test_conditional_branches():
    """式8-10 两种分支."""
    cond = ConditionalLoss(alpha=0.9, beta=0.1)
    # 分支1: L_CPAM < L_CE -> 加权
    l1 = cond(torch.tensor(1.0), torch.tensor(0.5))
    assert l1.item() > 0, "加权分支应为正"
    # 分支2: L_CPAM >= L_CE -> L_CE + beta*L_CPAM
    l2 = cond(torch.tensor(0.5), torch.tensor(1.0))
    expected = 0.5 + 0.1 * 1.0
    assert abs(l2.item() - expected) < 1e-5, f"{l2.item()} != {expected}"
    print(f"[OK] 分支1(加权)={l1.item():.4f}, 分支2(兜底)={l2.item():.4f}")


def test_conditional_symmetry():
    """α=0.9 时 W_CE 计算正确."""
    cond = ConditionalLoss(alpha=0.9, beta=0.1)
    lce, lcpam = torch.tensor(1.0), torch.tensor(0.5)
    # 手工验证加权
    alpha = 0.9
    w_ce = alpha * lce + (1 - alpha) * (lce / (lce + lcpam + 1e-8))
    w_cpam = alpha * lcpam + (1 - alpha) * (lcpam / (lce + lcpam + 1e-8))
    expected = (w_ce * lce + w_cpam * lcpam).item()
    got = cond(lce, lcpam).item()
    assert abs(got - expected) < 1e-5
    print(f"[OK] 加权计算: {got:.4f} ≈ {expected:.4f}")


def test_focal_posweight():
    """focal + pos_weight 维度与方向正确."""
    foc = FocalBCE(gamma=0.5, pos_weight=[2.0] * 3)
    logits = torch.randn(4, 3)
    targets = (torch.rand(4, 3) > 0.5).float()
    loss = foc(logits, targets)
    assert loss.dim() == 0, f"应标量: {loss.shape}"
    print(f"[OK] focal loss={loss.item():.4f}")


def test_aux_focal_loss_with_aux():
    """组合损失: 主头 + CPAM 辅助头."""
    loss_fn = AuxFocalLoss(num_classes=5, pos_weight=[1.0] * 5,
                            gamma=0.5, l2sp_weight=0.0)
    logits = torch.randn(4, 5)
    aux = torch.randn(4, 5)
    targets = (torch.rand(4, 5) > 0.6).float()
    total, d = loss_fn(logits, targets, aux_logits=aux)
    assert total.dim() == 0
    assert "L_CE" in d and "L_CPAM" in d and "total" in d
    print(f"[OK] total={total.item():.4f}, dict keys={list(d.keys())}")


def test_aux_focal_loss_no_aux():
    """无辅助头时退化为纯 L_CE."""
    loss_fn = AuxFocalLoss(num_classes=5, pos_weight=None, gamma=0.0, l2sp_weight=0.0)
    logits = torch.randn(4, 5)
    targets = (torch.rand(4, 5) > 0.6).float()
    total, d = loss_fn(logits, targets, aux_logits=None)
    assert total.dim() == 0
    assert "L_CPAM" not in d
    print(f"[OK] 无辅助: total={total.item():.4f}")


if __name__ == "__main__":
    test_conditional_branches()
    test_conditional_symmetry()
    test_focal_posweight()
    test_aux_focal_loss_with_aux()
    test_aux_focal_loss_no_aux()
    print("损失模块全部测试通过")