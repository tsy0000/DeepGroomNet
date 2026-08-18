"""losses 包: 行为分类损失 (自研命名).

- classification.FocalLossBCE / L2SPRegularizer: 多标签分类损失
- auxiliary.ConditionalLoss / AuxFocalLoss: CPAM 条件自适应损失 (式8-10)
"""
from .classification import FocalLossBCE, L2SPRegularizer, ClassificationLoss, build_loss
from .auxiliary import ConditionalLoss, AuxFocalLoss, FocalBCE

__all__ = ["FocalLossBCE", "L2SPRegularizer", "ClassificationLoss", "build_loss",
           "ConditionalLoss", "AuxFocalLoss", "FocalBCE"]