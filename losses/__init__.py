"""losses 包: DeepEthogram 官方多标签行为分类损失.

- auxiliary.FocalBCE: weighted binary focal loss (γ=1, pos_weight=(neg/pos)^0.25)
- auxiliary.ConditionalLoss: CPAM 条件自适应加权 (论文式8-10, α=0.9, β=0.1)
- auxiliary.L2SPRegularizer: L2-SP 向预训练权重衰减
- auxiliary.AuxFocalLoss: 主头 Focal + CPAM 条件辅助 + L2-SP 组合损失
"""
from .auxiliary import ConditionalLoss, AuxFocalLoss, FocalBCE, L2SPRegularizer

__all__ = ["ConditionalLoss", "AuxFocalLoss", "FocalBCE", "L2SPRegularizer"]