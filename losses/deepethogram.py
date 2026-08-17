"""DeepGroomNet_v3 - 损失函数（完全对齐 DeepEthogram 官方实现）。

- BinaryFocalLoss: pos_weight BCE + focal(γ) + label_smoothing*2 + ignore_index mask
- L2SPRegularizer: L2-SP 向预训练权重衰减（只对 2D 权重，α/β）
- ClassificationLoss: data_loss + reg_loss

源码依据: deepethogram/feature_extractor/losses.py
"""
import numpy as np
import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """DeepEthogram BinaryFocalLoss（多标签 BCE + focal + label smoothing）。"""

    def __init__(self, pos_weight=None, ignore_index=-1, gamma=0.0, label_smoothing=0.0,
                 exclusive_pairs=None, exclusive_weight=0.0):
        super().__init__()
        self.bcewithlogitsloss = nn.BCEWithLogitsLoss(weight=None, reduction="none",
                                                      pos_weight=pos_weight)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.eps = 1e-7
        self.label_smoothing = label_smoothing * 2
        # FG/BG 互斥正则 (Kalueff 梳理句法: 两种理毛不共存)。
        # config 里 binary_exclusivity_pairs=[2,3] / binary_exclusivity_weight, 之前配了但没生效。
        self.exclusive_pairs = exclusive_pairs or []
        self.exclusive_weight = exclusive_weight

    def forward(self, outputs, label):
        if outputs.shape != label.shape:
            if (1, *label.shape) == outputs.shape:
                label = label.unsqueeze(0)
        assert outputs.shape == label.shape
        label = label.float()
        N, K = outputs.shape
        # mask ignore_index (视频首尾 padding)
        mask = 1 - (label == self.ignore_index).to(torch.float).to(outputs.device)
        prob = torch.sigmoid(outputs)
        prob = torch.clamp(prob, self.eps, 1 - self.eps)
        # focal weight
        weight = (torch.pow(1 - prob, self.gamma) * label * mask
                  + torch.pow(prob, self.gamma) * (1 - label) * mask)
        # label smoothing
        label = label * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bceloss = self.bcewithlogitsloss(outputs, label)
        loss = (bceloss * weight).sum(dim=1).mean()
        # FG/BG 互斥正则: softplus(p_i * p_j) 惩罚两种理毛同时激活
        if self.exclusive_weight > 0 and self.exclusive_pairs:
            p = torch.sigmoid(outputs)
            excl = torch.zeros((), device=outputs.device)
            for (i, j) in self.exclusive_pairs:
                excl = excl + torch.nn.functional.softplus(p[:, i] * p[:, j]).mean()
            loss = loss + self.exclusive_weight * excl
        return loss

    @staticmethod
    def make_pos_weight(class_counts, power=0.25):
        """pos_weight = (num_neg/num_pos)^power (DeepEthogram make_loss_weight)。"""
        cc = np.asarray(class_counts, dtype=np.float64)
        pos = cc
        neg = cc.sum() - cc
        pw = (neg / np.clip(pos, 1, None)).clip(max=100) ** power
        return torch.from_numpy(pw).float()


class L2SPRegularizer(nn.Module):
    """L2-SP 向预训练权重衰减 (DeepEthogram regularization/style=l2_sp)。
    只对 ndim==2 权重: 预训练层用 α, 新层用 β。排除 bias/BN(ndim==1)。"""

    def __init__(self, model, alpha=1e-5, beta=1e-3, pretrained_keys=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self._pretrained_keys = set(pretrained_keys or [])
        with torch.no_grad():
            self._snapshot = {
                k: v.detach().clone().cpu()
                for k, v in model.state_dict().items()
                if v.ndim == 2 and v.dtype.is_floating_point
            }

    def forward(self, model):
        loss = torch.zeros((), device=next(model.parameters()).device)
        state = model.state_dict()
        for k, w0 in self._snapshot.items():
            w = state.get(k)
            if w is None or w.shape != w0.shape:
                continue
            pen = (w - w0.to(w.device)).pow(2).sum()
            coef = self.alpha if k in self._pretrained_keys else self.beta
            loss = loss + coef * pen
        return loss


class ClassificationLoss(nn.Module):
    """data_loss + reg_loss (DeepEthogram ClassificationLoss)。"""

    def __init__(self, data_criterion, reg_criterion):
        super().__init__()
        self.data_criterion = data_criterion
        self.reg_criterion = reg_criterion

    def forward(self, outputs, label, model):
        data_loss = self.data_criterion(outputs, label)
        reg_loss = self.reg_criterion(model)
        loss = data_loss + reg_loss
        if loss < 0 or loss != loss or torch.isinf(loss).sum() > 0:
            raise ValueError("invalid loss! Use anomaly detection")
        return loss, {"data_loss": data_loss.detach(), "reg_loss": reg_loss.detach()}


def build_loss(cfg, class_counts):
    """按 config 构建总损失 (DeepEthogram ClassificationLoss)。"""
    L = cfg["LOSS"]
    pw = BinaryFocalLoss.make_pos_weight(class_counts,
                                         power=float(L.get("binary_alpha_power", 0.25)))
    data_criterion = BinaryFocalLoss(
        pos_weight=pw,
        gamma=float(L.get("binary_gamma", 0.0)),
        label_smoothing=float(L.get("binary_label_smoothing", 0.05)),
    )
    return data_criterion  # 回归 L2-SP 在模型构建后单独加
