"""
模块C（Solution 3）：Focal Loss —— 解决类别极端不平衡。

实测类别分布（openfield，42 只小鼠帧级）：
  background 56.7% / Rear 21.5% / Locomote 18.7% / FaceGroom 2.2% / BodyGroom 0.67%

参考：Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
标准实现，支持：
  - 类别频率加权 alpha（mode: inverse | sqrt，提升稀有类 recall 且可报告权重）
  - label smoothing（平滑版 focal，可选）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        """
        alpha: 类别权重。可传 tensor([w0,...,wC-1])；或 "balanced" 由类频数自动算。
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self._alpha = alpha

    @staticmethod
    def balanced_alpha(class_counts: torch.Tensor, mode="inverse", min_alpha=None) -> torch.Tensor:
        """类别频数倒数归一化的权重，稀有类权重大。

        mode:
          "inverse" —— 1/count 归一化（稀有类权重极大，BG 1.7% 时 ~60x，易过拟合）
          "sqrt"    —— 1/sqrt(count) 归一化（缓解极不平衡的过拟合，稀有类权重温和放大）
          "deepethogram" —— (neg/pos)^β（DeepEthogram, eLife 2021）：β=0.25、pos=c 类
            频数、neg=其余频数（论文原文 "if only 1% positive, β=1 → weight 100" 即为
            neg/pos）。幂加权比反频温和得多：FG 2.2% 时权重 ~2.6 vs 反频归一化后同量级但
            平滑；背景 56.7% 时仅 ~0.93（不压死最大类）。作者实测 β=1 假阳性爆炸、
            β=0 假阴性过多，β=0.25 最优。
        min_alpha: 权重下界（默认 0.05）。sqrt 模式下 background 频数 29587 时
          权重会被压到 ~0.015 —— 把最大类完全压死（实测 background recall 掉到 14.5%）。
          设下界后重归一化，保住最大类的可学性（发表级：参数须报告）。
        """
        n = class_counts.float() + 1e-8
        if mode == "deepethogram":
            beta = 0.25
            a = ((n.sum() - n) / n).pow(beta)
        else:
            inv = 1.0 / n
            if mode == "sqrt":
                inv = inv.sqrt()
            a = inv / inv.sum()
        if min_alpha is not None:
            a = a.clamp(min=float(min_alpha))
            a = a / a.sum()
        return a

    @staticmethod
    def rare_class_bias(linear, class_counts, max_abs=3.0):
        """稀有类偏置初始化（DeepEthogram/RetinaNet 式）：b = -log((1-π)/π)。

        用类别经验先验 π 初始化 head 偏置 → 模型起点即匹配数据分布：
        常见类（background 56.7%）起手就有强负偏置，稀有类不被常见类先验淹没。
        对 nn.Linear 的 bias 就地修改；bias 为 None 时跳过。

        max_abs 实测需温和（默认 3.0 而非 10.0）：极端负偏置（-6.9）下稀有类需
        logit 爬升 ~7 个单位才过 0.5 阈，head-only 预热阶段（~6 epoch）内学不动，
        表现为"全空预测"（val 元素 acc=7/8 恒定、f1_macro=0、train argmax acc=多数类占比）。
        ±3.0 保留"常见类正先验、稀有类温和负先验"的语义，同时 logit 爬升距离减半。
        """
        from torch import nn as _nn
        if not isinstance(linear, _nn.Linear) or linear.bias is None:
            return
        pi = (class_counts.float() + 1e-8) / (class_counts.float() + 1e-8).sum()
        b = -torch.log((1.0 - pi) / (pi + 1e-8))
        linear.bias.data.copy_(b.clamp(-max_abs, max_abs))

    @property
    def alpha_values(self):
        """已固定的 alpha（用于训练日志打印/报告）。"""
        a = self._alpha
        if torch.is_tensor(a):
            return a.detach().cpu().tolist()
        return a

    def forward(self, logits, targets, class_counts=None):
        alpha = self._alpha
        if alpha == "balanced" or (alpha is None and class_counts is not None):
            if class_counts is None:
                raise ValueError("alpha='balanced' 需要提供 class_counts")
            alpha = self.balanced_alpha(class_counts)
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        if alpha is not None:
            alpha = alpha.to(device=logits.device, dtype=logits.dtype)

        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        ce = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1).detach()

        if self.label_smoothing > 0:
            pt = pt * (1.0 - self.label_smoothing) + self.label_smoothing / logits.size(1)

        loss = (1 - pt).pow(self.gamma) * ce
        if alpha is not None:
            a = alpha[targets]
            loss = loss * a
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _cfg_get(cfg, key, default):
    """兼容 dict 与 obj 两种配置容器（train.py 传 dict）。"""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class BinaryFocalLoss(nn.Module):
    """DeepEthogram 式多标签头损失（eLife 2021 原文方案）。

    原文：每个行为一个独立 sigmoid 分类器 + 验证集逐类阈值；背景=其余行为的逻辑非，
    不把背景当训练类。本实现：逐类 sigmoid BCE（可选 focal γ，Lin et al. ICCV 2017）
    + 逐类权重 alpha（默认 (neg/pos)^0.25，DeepEthogram 原文加权式）。
    γ=0 且 alpha=None 时退化为普通 BCE —— 最稳健、最贴近原文。
    """

    def __init__(self, gamma=1.0, alpha=None, reduction="mean", eval_threshold=0.5,
                 label_smoothing=0.0, exclusivity_pairs=None, exclusivity_weight=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.alpha_values = alpha.tolist() if alpha is not None else None
        self._eval_threshold = float(eval_threshold)
        self.label_smoothing = float(label_smoothing)
        self.exclusivity_pairs = exclusivity_pairs or []
        self.exclusivity_weight = float(exclusivity_weight)

    def forward(self, logits, targets):
        t = targets
        if self.label_smoothing > 0:
            sm = self.label_smoothing
            t = targets * (1.0 - sm) + sm * 0.5 * (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        if self.gamma > 0:
            pt = torch.exp(-bce)
            bce = (1.0 - pt) ** self.gamma * bce
        if self.alpha_values is not None:
            a = torch.as_tensor(self.alpha_values, dtype=logits.dtype, device=logits.device)
            bce = bce * a.view(1, -1)
        if self.reduction == "mean":
            loss = bce.mean()
        elif self.reduction == "sum":
            loss = bce.sum()
        else:
            return bce
        if self.exclusivity_weight > 0 and self.exclusivity_pairs:
            p = torch.sigmoid(logits.float())
            excl = torch.zeros((), dtype=logits.dtype, device=logits.device)
            n = 0
            for i, j in self.exclusivity_pairs:
                excl = excl + F.softplus(p[:, i] * p[:, j]).mean()
                n += 1
            if n > 0:
                loss = loss + self.exclusivity_weight * (excl / n)
        return loss


def build_loss(cfg, class_counts=None):
    """按 config 构建损失：focal 或带类别权重的 CE。
    类别权重在构建时固定（用训练集统计一次），避免推理/验证时再传 class_counts。
    config.LOSS.focal_alpha_mode: inverse | sqrt | deepethogram（见 FocalLoss.balanced_alpha）。
    head_mode=multilabel（默认）：DeepEthogram 式逐类 sigmoid 头（背景=补集，稳健方案）。
    """
    if _cfg_get(cfg, "head_mode", "multilabel") == "multilabel":
        alpha = None
        if class_counts is not None:
            alpha = FocalLoss.balanced_alpha(
                class_counts, mode=_cfg_get(cfg, "binary_alpha_mode", "deepethogram"))
        excl = _cfg_get(cfg, "binary_exclusivity_pairs", None)
        return BinaryFocalLoss(gamma=_cfg_get(cfg, "binary_gamma", 1.0), alpha=alpha,
                               eval_threshold=_cfg_get(cfg, "binary_eval_threshold", 0.5),
                               label_smoothing=_cfg_get(cfg, "binary_label_smoothing", 0.0),
                               exclusivity_pairs=excl,
                               exclusivity_weight=_cfg_get(cfg, "binary_exclusivity_weight", 0.0))
    use_focal = _cfg_get(cfg, "use_focal", True)
    if use_focal:
        alpha = None
        if _cfg_get(cfg, "focal_alpha_balanced", True):
            if class_counts is None:
                raise ValueError("use_focal 且 focal_alpha_balanced=True 时需要提供 class_counts")
            mode = _cfg_get(cfg, "focal_alpha_mode", "inverse")
            min_a = _cfg_get(cfg, "focal_alpha_min", 0.05)
            alpha = FocalLoss.balanced_alpha(class_counts, mode=mode, min_alpha=min_a)
        return FocalLoss(
            gamma=_cfg_get(cfg, "focal_gamma", 2.0),
            alpha=alpha,
            label_smoothing=_cfg_get(cfg, "label_smoothing", 0.1),
        )
    w = None
    if class_counts is not None:
        w = FocalLoss.balanced_alpha(class_counts)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=_cfg_get(cfg, "label_smoothing", 0.1))
