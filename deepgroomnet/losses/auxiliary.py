import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalLoss(nn.Module):

    def __init__(self, alpha=0.9, beta=0.1, eps=1e-08):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, lce, lcpam):
        if lcpam < lce:
            d_ce = lce.detach()
            d_cp = lcpam.detach()
            w_ce = self.alpha * d_ce + (1 - self.alpha) * (d_ce / (d_ce + d_cp + self.eps))
            w_cpam = self.alpha * d_cp + (1 - self.alpha) * (d_cp / (d_ce + d_cp + self.eps))
            return w_ce * lce + w_cpam * lcpam
        else:
            return lce + self.beta * lcpam

class FocalBCE(nn.Module):

    def __init__(self, gamma=0.5, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if pos_weight is not None:
            self.pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        pt = torch.clamp(pt, min=1e-07, max=1.0)
        focal = (1 - pt) ** self.gamma
        loss = F.binary_cross_entropy(probs.clamp(1e-07, 1 - 1e-07), targets, reduction='none')
        loss = focal * loss
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device).view(1, -1)
            loss = loss * (1 + (pw - 1) * targets)
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class L2SPRegularizer(nn.Module):

    def __init__(self, weight=0.001, ignore_names=('head', 'cpam', 'stsa')):
        super().__init__()
        self.weight = weight
        self.ignore_names = ignore_names

    def forward(self, model):
        reg = None
        for (name, p) in model.named_parameters():
            if p.requires_grad and (not any((k in name for k in self.ignore_names))):
                reg = reg + torch.sum(p ** 2) if reg is not None else torch.sum(p ** 2)
        if reg is None:
            return 0.0
        return self.weight * reg

class AuxFocalLoss(nn.Module):

    def __init__(self, num_classes, pos_weight=None, gamma=1.0, alpha=0.9, beta=0.1, l2sp_weight=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.focal = FocalBCE(gamma=gamma, pos_weight=pos_weight)
        self.conditional = ConditionalLoss(alpha=alpha, beta=beta)
        self.l2sp = L2SPRegularizer(weight=l2sp_weight)

    def forward(self, logits, targets, aux_logits=None, model=None):
        tgt = targets.float()
        lce = self.focal(logits.float(), tgt)
        loss_dict = {'L_CE': lce.detach()}
        if aux_logits is not None:
            lcpam = self.focal(aux_logits.float(), tgt)
            loss_dict['L_CPAM'] = lcpam.detach()
            data_loss = self.conditional(lce, lcpam)
            loss_dict['combined'] = data_loss.detach()
        else:
            data_loss = lce
        if model is not None:
            l2 = self.l2sp(model)
            l2 = torch.as_tensor(l2, dtype=torch.float32, device=logits.device)
            loss_dict['L2-SP'] = l2.detach()
        else:
            l2 = torch.zeros((), dtype=torch.float32, device=logits.device)
        total = data_loss + l2
        loss_dict['total'] = total.detach()
        return (total, loss_dict)
