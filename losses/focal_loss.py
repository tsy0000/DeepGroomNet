import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self._alpha = alpha

    @staticmethod
    def balanced_alpha(class_counts: torch.Tensor, mode='inverse', min_alpha=None) -> torch.Tensor:
        n = class_counts.float() + 1e-08
        if mode == 'deepethogram':
            beta = 0.25
            a = ((n.sum() - n) / n).pow(beta)
        else:
            inv = 1.0 / n
            if mode == 'sqrt':
                inv = inv.sqrt()
            a = inv / inv.sum()
        if min_alpha is not None:
            a = a.clamp(min=float(min_alpha))
            a = a / a.sum()
        return a

    @staticmethod
    def rare_class_bias(linear, class_counts, max_abs=10.0):
        from torch import nn as _nn
        if not isinstance(linear, _nn.Linear) or linear.bias is None:
            return
        pi = (class_counts.float() + 1e-08) / (class_counts.float() + 1e-08).sum()
        b = -torch.log((1.0 - pi) / (pi + 1e-08))
        linear.bias.data.copy_(b.clamp(-max_abs, max_abs))

    @property
    def alpha_values(self):
        a = self._alpha
        if torch.is_tensor(a):
            return a.detach().cpu().tolist()
        return a

    def forward(self, logits, targets, class_counts=None):
        alpha = self._alpha
        if alpha == 'balanced' or (alpha is None and class_counts is not None):
            if class_counts is None:
                raise ValueError("alpha='balanced' 需要提供 class_counts")
            alpha = self.balanced_alpha(class_counts)
        if alpha is not None and (not torch.is_tensor(alpha)):
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
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def _cfg_get(cfg, key, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=1.0, alpha=None, reduction='mean', eval_threshold=0.5):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.alpha_values = alpha.tolist() if alpha is not None else None
        self._eval_threshold = float(eval_threshold)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.gamma > 0:
            pt = torch.exp(-bce)
            bce = (1.0 - pt) ** self.gamma * bce
        if self.alpha_values is not None:
            a = torch.as_tensor(self.alpha_values, dtype=logits.dtype, device=logits.device)
            bce = bce * a.view(1, -1)
        if self.reduction == 'mean':
            return bce.mean()
        if self.reduction == 'sum':
            return bce.sum()
        return bce

def build_loss(cfg, class_counts=None):
    if _cfg_get(cfg, 'head_mode', 'multilabel') == 'multilabel':
        alpha = None
        if class_counts is not None:
            alpha = FocalLoss.balanced_alpha(class_counts, mode=_cfg_get(cfg, 'binary_alpha_mode', 'deepethogram'))
        return BinaryFocalLoss(gamma=_cfg_get(cfg, 'binary_gamma', 1.0), alpha=alpha, eval_threshold=_cfg_get(cfg, 'binary_eval_threshold', 0.5))
    use_focal = _cfg_get(cfg, 'use_focal', True)
    if use_focal:
        alpha = None
        if _cfg_get(cfg, 'focal_alpha_balanced', True):
            if class_counts is None:
                raise ValueError('use_focal 且 focal_alpha_balanced=True 时需要提供 class_counts')
            mode = _cfg_get(cfg, 'focal_alpha_mode', 'inverse')
            min_a = _cfg_get(cfg, 'focal_alpha_min', 0.05)
            alpha = FocalLoss.balanced_alpha(class_counts, mode=mode, min_alpha=min_a)
        return FocalLoss(gamma=_cfg_get(cfg, 'focal_gamma', 2.0), alpha=alpha, label_smoothing=_cfg_get(cfg, 'label_smoothing', 0.1))
    w = None
    if class_counts is not None:
        w = FocalLoss.balanced_alpha(class_counts)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=_cfg_get(cfg, 'label_smoothing', 0.1))
