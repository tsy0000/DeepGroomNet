"""
条件自适应损失 (论文式 8-10) + 组合损失。

论文 (Swin-Panda, Diversity 2025) 式 8-10:
  L_total = W_CE·L_CE + W_CPAM·L_CPAM   if L_CPAM < L_CE
  L_total = L_CE + β·L_CPAM              otherwise
  W_CE   = α·L_CE + (1-α)·L_CE/(L_CE+L_CPAM)
  W_CPAM = α·L_CPAM + (1-α)·L_CPAM/(L_CE+L_CPAM)

最优参数: α=0.9, β=0.1 (论文 4.3.4 消融验证).

本模块还提供组合损失: 主头 BCE(focal) + CPAM 条件辅助 + L2-SP 正则。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLoss(nn.Module):
    """式 8-10 条件自适应加权损失.

    Args:
        alpha (float): 加权超参, 默认 0.9.
        beta (float): 兜底权重, 默认 0.1.
        eps (float): 数值稳定.
    """

    def __init__(self, alpha=0.9, beta=0.1, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, lce, lcpam):
        """lce, lcpam: 标量 loss. 权重用 detach 值计算 (式9/10), 梯度只经损失项回传.

        论文式8-10: 权重 W_CE/W_CPAM 由 L_CE/L_CPAM 计算, 应视为常数 (stop-gradient),
        梯度仅通过 L_CE/L_CPAM 本身回传, 避免权重额外引入梯度 (与论文公式一致).
        """
        if lcpam < lce:
            d_ce = lce.detach(); d_cp = lcpam.detach()
            w_ce = self.alpha * d_ce + (1 - self.alpha) * (d_ce / (d_ce + d_cp + self.eps))
            w_cpam = self.alpha * d_cp + (1 - self.alpha) * (d_cp / (d_ce + d_cp + self.eps))
            return w_ce * lce + w_cpam * lcpam
        else:
            return lce + self.beta * lcpam


class FocalBCE(nn.Module):
    """Focal 二分类损失 (Lin ICCV17) + pos_weight, 用于多标签行为分类.

    应对极度不平衡 (FG 4.3% / BG 1.9%).
    """

    def __init__(self, gamma=0.5, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # pos_weight: (num_classes,) tensor 或 list/None
        if pos_weight is not None:
            self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        """logits: (N, C) 未激活; targets: (N, C) 0/1."""
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)   # p_t
        pt = torch.clamp(pt, min=1e-7, max=1.0)              # 数值稳定

        # focal 调制
        focal = (1 - pt) ** self.gamma

        loss = F.binary_cross_entropy(probs.clamp(1e-7, 1 - 1e-7), targets,
                                      reduction="none")
        loss = focal * loss

        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device).view(1, -1)
            loss = loss * (1 + (pw - 1) * targets)           # 正样本加权

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class L2SPRegularizer(nn.Module):
    """L2-SP 正则: 约束主干参数向预训练权重 (迁移学习防偏离)."""

    def __init__(self, weight=1e-3, ignore_names=("head", "cpam", "stsa")):
        super().__init__()
        self.weight = weight
        self.ignore_names = ignore_names

    def forward(self, model):
        reg = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and not any(k in name for k in self.ignore_names):
                reg = reg + torch.sum(p ** 2)
        return self.weight * reg


class TadaPandaLoss(nn.Module):
    """组合损失: 主头 FocalBCE + CPAM 条件辅助 (式8-10) + L2-SP.

    Args:
        num_classes (int): 分类数.
        pos_weight (list): 类别正样本权重.
        gamma (float): focal gamma.
        alpha, beta (float): 条件损失参数.
        l2sp_weight (float): L2-SP 正则权重.
        aux_weight (float): 辅助分支是否恒有贡献 (内部条件化).
    """

    def __init__(self, num_classes, pos_weight=None, gamma=0.5,
                 alpha=0.9, beta=0.1, l2sp_weight=1e-3, label_smoothing=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.focal = FocalBCE(gamma=gamma, pos_weight=pos_weight)
        self.conditional = ConditionalLoss(alpha=alpha, beta=beta)
        self.l2sp = L2SPRegularizer(weight=l2sp_weight)
        self.label_smoothing = label_smoothing

    def _smooth(self, targets):
        """label smoothing for multi-label."""
        if self.label_smoothing > 0:
            return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return targets

    def forward(self, logits, targets, aux_logits=None, model=None):
        """logits: 主头 (N,C); targets: (N,C) one-hot; aux_logits: CPAM 头 or None.

        返回 (loss, dict{name: value})。
        """
        tgt = self._smooth(targets.float())
        lce = self.focal(logits.float(), tgt)

        loss_dict = {"L_CE": lce.detach()}

        if aux_logits is not None:
            lcpam = self.focal(aux_logits.float(), tgt)
            loss_dict["L_CPAM"] = lcpam.detach()
            # 式8-10 条件自适应加权 (ConditionalLoss 内部对权重 detach, 梯度仅经损失项)
            data_loss = self.conditional(lce, lcpam)
            loss_dict["combined"] = data_loss.detach()
        else:
            data_loss = lce

        if model is not None:
            l2 = self.l2sp(model)
            loss_dict["L2-SP"] = l2.detach()
        else:
            l2 = 0.0

        total = data_loss + l2
        loss_dict["total"] = total.detach()
        return total, loss_dict