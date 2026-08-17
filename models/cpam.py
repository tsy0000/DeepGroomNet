"""
CPAM - Comprehensive Perception Auxiliary Module（严格按 Swin-Panda, Diversity 2025 原文）。

论文结构（式2-7）:
  Xi (N×Ci×T×H×W, i=1..4) 各线性投影 fi -> Pi
  P̂i = 插值(Pi, size=shape(P4))                       (式3)
  Mi = gi(P̂i)  (mixer)                                 (式4)
  X  = Reshape(Concat([Mi], dim=1))                    (式5)
  A  = Softmax(MLP(GAP(X))),  A ∈ N×4×C×1×1           (式6, 逐stage逐通道融合权重)
  Y  = Σ_{i=1}^4 Ai·Xi                                 (式7)
  Y 进辅助分类头 -> 辅助损失 L_CPAM

论文辅助损失自适应（式8-10, α=0.9, β=0.1）:
  L_total = W_CE·L_CE + W_CPAM·L_CPAM   if L_CPAM < L_CE
  L_total = L_CE + β·L_CPAM              otherwise
  W_CE  = α·L_CE + (1-α)·L_CE/(L_CE+L_CPAM)
  W_CPAM= α·L_CPAM + (1-α)·L_CPAM/(L_CE+L_CPAM)

注: Swin-Panda 未开源, 本实现严格依据论文描述。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPAM(nn.Module):
    def __init__(self, stage_channels, out_ch, num_classes, interpolate_size=(16, 7, 7),
                 dropout=0.1):
        super().__init__()
        self.num_stages = len(stage_channels)
        self.interpolate_size = interpolate_size
        # fi: 线性投影, 各 stage 通道 -> out_ch
        self.proj = nn.ModuleList([
            nn.Conv3d(c, out_ch, kernel_size=1, bias=False) for c in stage_channels
        ])
        # gi: mixer 层
        self.mixers = nn.ModuleList([
            nn.Sequential(nn.Conv3d(out_ch, out_ch, kernel_size=1), nn.GELU())
            for _ in stage_channels
        ])
        # 融合权重: GAP(X [N,4,C,T,H,W]) -> [N,4,C] -> MLP -> [N,4,C] -> softmax(沿4)
        self.fusion = nn.Sequential(
            nn.Linear(out_ch * self.num_stages, out_ch * self.num_stages), nn.ReLU(inplace=True),
            nn.Linear(out_ch * self.num_stages, out_ch * self.num_stages),
        )
        # 辅助分类头
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(1),
            nn.Dropout(dropout), nn.Linear(out_ch, num_classes),
        )

    def forward(self, feats):
        """feats: list of 4 stage features [N,Ci,T,H,W]。返回 (Y [N,C,T,H,W], aux_logits [N,C])。"""
        N = feats[0].shape[0]
        proj = []
        for i, x in enumerate(feats):
            p = self.proj[i](x)                          # [N,C,T,H,W]
            p = F.interpolate(p, size=self.interpolate_size, mode="trilinear", align_corners=False)
            p = self.mixers[i](p)
            proj.append(p)
        X = torch.stack(proj, dim=1)                     # [N,4,C,T,H,W]
        # 式6: A = Softmax(MLP(GAP(X))), 逐stage逐通道权重
        g = X.mean(dim=(3, 4, 5))                        # GAP -> [N,4,C]
        A = F.softmax(self.fusion(g.reshape(N, -1)).view(N, self.num_stages, -1), dim=1)  # [N,4,C]
        # 式7: Y = Σ Ai·Xi
        Y = (X * A[:, :, :, None, None, None]).sum(dim=1)   # [N,C,T,H,W]
        aux_logits = self.aux_head(Y)                    # [N,num_classes]
        return Y, aux_logits


def conditional_loss(lce, lcpam, alpha=0.9, beta=0.1):
    """式8-10: 条件自适应加权 L_CE 与 L_CPAM。lce/lcpam: 标量 tensor。"""
    if lcpam < lce:
        w_ce = alpha * lce + (1 - alpha) * (lce / (lce + lcpam + 1e-8))
        w_cpam = alpha * lcpam + (1 - alpha) * (lcpam / (lce + lcpam + 1e-8))
        return w_ce * lce + w_cpam * lcpam
    else:
        return lce + beta * lcpam
