"""
关键点分支（PoseBranch）v2：PoseC3D 式热图 3D CNN 流（mmaction2 主流的骨架建模）。

对照旧版（坐标 + MLP）的升级：
  - 输入：PoseTargetGenerator 生成的伪热图 [T, C, H, W]（关键点高斯 + 肢体线段，PoseC3D 原版）
  - 结构：3D 卷积栈（参考 PoseC3D 的 stem 结构，简单 3 层），输出与视觉 logits 晚融合
  - 热图流对"新视频外观差异"鲁棒：热图只编码关键点几何，与光照/颜色/背景无关

参考：Duan et al., "Revisiting Skeleton-based Action Recognition", CVPR 2021 (PoseC3D)；
     mmaction2 实现（open-mmlab/mmaction2）。
"""
import torch
import torch.nn as nn


class PoseBranch(nn.Module):
    """热图流 3D CNN（PoseC3D 风格）：[B,T,C,H,W] -> [B, num_classes]。"""

    def __init__(self, in_channels, num_classes, heatmap_size=64,
                 hidden=128, dropout=0.4):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        for m in self.conv:
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, heatmaps):
        """heatmaps: [B, T, C, H, W]（PoseTargetGenerator.gen 输出 [T,C,H,W]，batch 在前）。"""
        with torch.cuda.amp.autocast(enabled=False):
            x = heatmaps.float().permute(0, 2, 1, 3, 4)
            return self.conv(x)
