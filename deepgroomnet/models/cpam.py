import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalMixer(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class _SKFusion3D(nn.Module):

    def __init__(self, dim, height=4, reduction=8):
        super().__init__()
        self.height = height
        d = max(int(dim // reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(nn.Conv3d(dim, d, 1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(d, dim * height, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        (B, height, C, T, H, W) = in_feats.shape
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, height, C, 1, 1, 1))
        out = torch.sum(in_feats * attn, dim=1)
        return out

class CPAM(nn.Module):

    def __init__(self, stage_channels, num_classes, out_ch=None, temporal_mixer=True):
        super().__init__()
        self.num_stages = len(stage_channels)
        self.out_ch = out_ch or stage_channels[-1]
        self.proj = nn.ModuleList([nn.Conv3d(c, self.out_ch, kernel_size=1, bias=False) for c in stage_channels])
        self.mixers = nn.ModuleList([TemporalMixer(self.out_ch) for _ in stage_channels]) if temporal_mixer else None
        self.fusion = _SKFusion3D(self.out_ch, height=self.num_stages, reduction=8)
        self.aux_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(1), nn.Dropout(0.1), nn.Linear(self.out_ch, num_classes))

    def forward(self, feats):
        target = feats[-1].shape[2:]
        proj = []
        for (i, x) in enumerate(feats):
            p = self.proj[i](x)
            if p.shape[2:] != tuple(target):
                p = F.interpolate(p, size=tuple(target), mode='trilinear', align_corners=False)
            if self.mixers is not None:
                p = self.mixers[i](p)
            proj.append(p)
        X = torch.stack(proj, dim=1)
        Y = self.fusion(X)
        aux_logits = self.aux_head(Y)
        return (Y, aux_logits)
