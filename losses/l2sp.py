"""
L2-SP 正则化（DeepEthogram eLife 2021 原版：L2-SP from the pretrained weights）。

动机：ImageNet/K400 预训练权重在小数据微调时易被破坏（灾难性遗忘）。
L2-SP = 惩罚"当前权重偏离预训练权重"，但只对"预训练过的层"生效：
  - 权重（dim>1）：按 α 惩罚 ‖θ - θ₀‖²
  - bias/BN（dim==1）：按 β 惩罚（β 通常大于 α，DeepEthogram: α=1e-5, β=1e-3）
新初始化的头/分支（aux_head、pose_branch、替换后的 head）不在预训练快照中 → 不惩罚。

参考：Li et al. "Explicit Inductive Bias for Transfer Learning with Convolutional
Networks" (ICML 2018)；DeepEthogram losses.py L2SPRegularizer 同语义。
"""
import torch
import torch.nn as nn


class L2SPRegularizer(nn.Module):
    """对预训练参数施加 L2-SP 惩罚。用法：
        model 构建+加载预训练后：l2sp = L2SPRegularizer(model)
        每 step: loss = task_loss + l2sp(model)
    """

    def __init__(self, model: nn.Module, alpha: float = 1e-5, beta: float = 1e-3,
                 exclude_prefixes=("head", "aux_head", "pose_branch")):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._exclude = tuple(exclude_prefixes)
        with torch.no_grad():
            self._snapshot = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if v.ndim in (1, 2, 3, 4, 5) and v.dtype.is_floating_point
                and not k.startswith(self._exclude)
            }
        self._names = list(self._snapshot.keys())

    def forward(self, model: nn.Module) -> torch.Tensor:
        """返回 Σ[α·‖W-W0‖² (权重) + β·‖b-b0‖² (bias/BN)]；与当前参数对齐匹配。"""
        loss = torch.zeros((), device=next(model.parameters()).device)
        state = model.state_dict()
        for k, w0 in self._snapshot.items():
            w = state.get(k)
            if w is None or w.shape != w0.shape:
                continue
            pen = (w - w0.to(w.device)).pow(2).sum()
            loss = loss + (self.beta if w.dim() == 1 else self.alpha) * pen
        return loss


def build_l2sp(cfg, model):
    """按 config 构建 L2-SP（LOSS.l2sp 段）；返回 nn.Module 或 None。"""
    l2 = (cfg or {}).get("l2sp", {}) if isinstance(cfg, dict) else {}
    if not l2.get("enabled", False):
        return None
    a = float(l2.get("alpha", 1e-5))
    b = float(l2.get("beta", 1e-3))
    return L2SPRegularizer(model, alpha=a, beta=b)
