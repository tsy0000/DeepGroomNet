"""
CPAM - Comprehensive Perception Auxiliary Module.

严格按 Swin-Panda 论文 (Diversity 2025, 17, 139) 式 2-7 实现，融合权重部分
参考 DehazeFormer (即论文引用 [21] 的开源实现) 的 SKFusion 逻辑。

论文结构:
  Xi (N×Ci×T×H×W, i=1..4) 各线性投影 fi -> Pi                      (式2)
  P̂i = 插值(Pi, size=shape(P4))                                     (式3)
  Mi = gi(P̂i)  (时间混合, (3,1,1) Conv3d)                            (式4)
  X  = stack([Mi], dim=1) -> (N,4,C,T,H,W)                          (式5)
  A  = Softmax(MLP(GAP(X)), dim=1), A ∈ (N,4,C,1,1,1)               (式6)
  Y  = Σ_{i=1..4} Ai·Xi                                             (式7)
  Y -> 辅助分类头 -> 辅助损失 L_CPAM

设计要点（与用户确认一致）:
  1. 输入是 4 个 STSA 之前的原始 stage 特征（严格按论文 Figure 1）。
  2. Temporal Mixer ((3,1,1) Conv3d) 在 SKFusion 之前、对每个尺度各做一次。
  3. 自适应融合权重用 DehazeFormer SKFusion 权威写法（GAP -> MLP -> softmax 尺度维）。
  4. 对齐目标 = 最深层 stage 的 (T,H,W)，TAdaConvNeXtV2 下 T 全程不变。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixer(nn.Module):
    """式4 gi: 对单尺度特征做帧间时序建模的时间混合器。(3,1,1) Conv3d + BN."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False
        )
        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _SKFusion3D(nn.Module):
    """式6: 自适应融合权重生成 + 式7 加权求和。

    迁移自 DehazeFormer SKFusion (models/dehazeformer.py, L349-376),
    从 2D 改为 3D。输入 stack (N, height, C, T, H, W)，输出 (N, C, T, H, W)。
    """

    def __init__(self, dim, height=4, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim // reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(d, dim * height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        # in_feats: (B, height, C, T, H, W)
        B, height, C, T, H, W = in_feats.shape
        feats_sum = torch.sum(in_feats, dim=1)                 # (B, C, T, H, W)
        attn = self.mlp(self.avg_pool(feats_sum))              # (B, height*C, 1,1,1)
        attn = self.softmax(attn.view(B, height, C, 1, 1, 1))  # 尺度维 softmax
        out = torch.sum(in_feats * attn, dim=1)                # (B, C, T, H, W)
        return out


class CPAM(nn.Module):
    """Comprehensive Perception Auxiliary Module.

    Args:
        stage_channels (list[int]): 各 stage 输出通道, 如 [96,192,384,768].
        num_classes (int): 分类数.
        out_ch (int): 统一投影通道 (默认取最深层通道 768).
        temporal_mixer (bool): 是否保留 (3,1,1) Temporal Mixer (式4). 默认 True.
    """

    def __init__(self, stage_channels, num_classes, out_ch=None, temporal_mixer=True):
        super().__init__()
        self.num_stages = len(stage_channels)
        self.out_ch = out_ch or stage_channels[-1]

        # 式2 fi: 线性投影, 各 stage 通道 -> out_ch
        self.proj = nn.ModuleList([
            nn.Conv3d(c, self.out_ch, kernel_size=1, bias=False) for c in stage_channels
        ])

        # 式4 gi: 时间混合 (对每尺度各一次, 在融合之前)
        self.mixers = nn.ModuleList([
            TemporalMixer(self.out_ch) for _ in stage_channels
        ]) if temporal_mixer else None

        # 式6/7: SKFusion 自适应加权融合
        self.fusion = _SKFusion3D(self.out_ch, height=self.num_stages, reduction=8)

        # 辅助分类头
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Dropout(0.1),
            nn.Linear(self.out_ch, num_classes),
        )

    def forward(self, feats):
        """feats: list of 4 stage features [N,Ci,T,H,W]。返回 (Y, aux_logits).

        Y: (N, out_ch, T, H, W) 融合特征; aux_logits: (N, num_classes)。
        """
        # 式2 投影 + 式3 插值对齐到最深层空间尺寸
        target = feats[-1].shape[2:]  # (T,H,W)
        proj = []
        for i, x in enumerate(feats):
            p = self.proj[i](x)
            if p.shape[2:] != tuple(target):
                p = F.interpolate(
                    p, size=tuple(target), mode="trilinear", align_corners=False
                )
            if self.mixers is not None:
                p = self.mixers[i](p)  # 式4
            proj.append(p)

        # 式5: 沿新尺度维 stack
        X = torch.stack(proj, dim=1)            # (N, num_stages, C, T, H, W)

        # 式6+式7: SKFusion 自适应加权融合
        Y = self.fusion(X)                      # (N, C, T, H, W)

        aux_logits = self.aux_head(Y)           # (N, num_classes)
        return Y, aux_logits