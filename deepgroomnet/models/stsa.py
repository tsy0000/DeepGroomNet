import torch
import torch.nn as nn
import torch.nn.functional as F

def spatial_shift1(x):
    (N, C, T, H, W) = x.shape
    (c1, c2, c3) = (C // 4, C // 2, 3 * C // 4)
    out = torch.zeros_like(x)
    out[:, :c1, :, 1:, :] = x[:, :c1, :, :H - 1, :]
    out[:, c1:c2, :, :H - 1, :] = x[:, c1:c2, :, 1:, :]
    out[:, c2:c3, :, :, 1:] = x[:, c2:c3, :, :, :W - 1]
    out[:, c3:, :, :, :W - 1] = x[:, c3:, :, :, 1:]
    return out

def spatial_shift2(x):
    (N, C, T, H, W) = x.shape
    (c1, c2, c3) = (C // 4, C // 2, 3 * C // 4)
    out = torch.zeros_like(x)
    out[:, :c1, :, :, 1:] = x[:, :c1, :, :, :W - 1]
    out[:, c1:c2, :, :, :W - 1] = x[:, c1:c2, :, :, 1:]
    out[:, c2:c3, :, 1:, :] = x[:, c2:c3, :, :H - 1, :]
    out[:, c3:, :, :H - 1, :] = x[:, c3:, :, 1:, :]
    return out

class SplitAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.groups = 3
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, feats):
        (N, C, T, H, W) = feats[0].shape
        x = torch.stack(feats, dim=1)
        s = x.sum(dim=(3, 4, 5))
        att = self.mlp(s)
        alpha = F.softmax(att, dim=1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = (x * alpha).sum(dim=1)
        return out

class TemporalAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        (N, C, T, H, W) = feats[0].shape
        x = torch.stack(feats, dim=1).sum(dim=1)
        diff = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        mask = self.sigmoid(self.conv(diff))
        return x * mask

class STSA(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        C = channels
        self.mlp1 = nn.Sequential(nn.Conv3d(C, C, kernel_size=1), nn.GELU(), nn.Conv3d(C, C * 3, kernel_size=1))
        self.sa = SplitAttention(channels)
        self.ta = TemporalAttention(channels)
        self.mlp2 = nn.Sequential(nn.Conv3d(C, C, kernel_size=1), nn.GELU(), nn.Conv3d(C, C, kernel_size=1))

    def forward(self, x):
        input = x
        chi = self.mlp1(x)
        C = self.channels
        x1 = chi[:, :C]
        x2 = chi[:, C:2 * C]
        x3 = chi[:, 2 * C:]
        x1 = spatial_shift1(x1)
        x2 = spatial_shift2(x2)
        out = self.sa([x1, x2, x3]) + self.ta([x1, x2, x3])
        out = self.mlp2(out)
        return input + out
