"""
STSA - Spatiotemporal Shift Attention Module.

严格按 Swin-Panda 论文 (Diversity 2025, 17, 139) 式 11-15 实现。
空间移位灵感来自 S2-MLPv2 [22] (arXiv 2108.01072)，改造为 3D 视频版本：
输入 (N,C,T,H,W)，移位只在 H/W 空间做、保留时间维 T 不变。

论文结构:
  χ̂  = MLP1(χ) ∈ R^{3C×T×H×W}                  (式11, CM-MLP: C->3C)
  χ1,χ2,χ3 = split(χ̂, 3, dim=C)                 (式12)
  χ1 <- SS1(χ1), χ2 <- SS2(χ2), χ3 不变          (式13/14 非对称空间移位)
  X  = MLP2( SA(χ̃) + TA(χ̃) )                  (式15, SA空间 + TA时间注意力)
  out = χ + X                                     (残差)

维度一致性 (修正此前 TA 分支 3C/C 不匹配):
  - SA: 输入 3 路 C 通道移位特征 -> 空间统计 -> MLP -> softmax(分组) -> 加权求和 (C)。
        (严格迁移 S2-MLPv2 [22] 的 split-attention, 式2-4, 而非纯 softmax 分支权重)
  - TA: 输入 3 路 C 通道 -> 帧差分 -> 时序卷积 -> softmax(时间维) -> 加权求和 (C)。
  - MLP1: C->3C;  MLP2: C->C (残差保持通道不变)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_shift1(x):
    """非对称空间移位 SS1 (3D 版, 仅 H/W)。

    方向: 通道块 [c/4|c/2|3c/4] 沿 H/W 平移, 边界补 0 (论文式13)。
    x: (N, C, T, H, W)
    """
    N, C, T, H, W = x.shape
    c1, c2, c3 = C // 4, C // 2, 3 * C // 4
    out = torch.zeros_like(x)
    # 0~c/4: 沿 H 下移
    out[:, :c1, :, 1:, :] = x[:, :c1, :, :H - 1, :]
    # c/4~c/2: 沿 H 上移
    out[:, c1:c2, :, :H - 1, :] = x[:, c1:c2, :, 1:, :]
    # c/2~c3/4: 沿 W 右移
    out[:, c2:c3, :, :, 1:] = x[:, c2:c3, :, :, :W - 1]
    # c3/4~C: 沿 W 左移
    out[:, c3:, :, :, :W - 1] = x[:, c3:, :, :, 1:]
    return out


def spatial_shift2(x):
    """非对称空间移位 SS2 (与 SS1 方向相反的组合, 论文式14)。"""
    N, C, T, H, W = x.shape
    c1, c2, c3 = C // 4, C // 2, 3 * C // 4
    out = torch.zeros_like(x)
    # 0~c/4: 沿 W 右移
    out[:, :c1, :, :, 1:] = x[:, :c1, :, :, :W - 1]
    # c/4~c/2: 沿 W 左移
    out[:, c1:c2, :, :, :W - 1] = x[:, c1:c2, :, :, 1:]
    # c/2~c3/4: 沿 H 下移
    out[:, c2:c3, :, 1:, :] = x[:, c2:c3, :, :H - 1, :]
    # c3/4~C: 沿 H 上移
    out[:, c3:, :, :H - 1, :] = x[:, c3:, :, 1:, :]
    return out


class SplitAttention(nn.Module):
    """SA 空间注意力分支 (论文 Figure 5a + S2-MLPv2 [22] split-attention 式2-4)。

    输入 3 路移位特征 (各 C 通道)，先求和得全局统计，经 MLP(Linear-ReLU-Linear)
    学到 3 组权重，softmax(分组维) 后对三路做加权求和 -> (N,C,T,H,W)。
    """

    def __init__(self, channels):
        super().__init__()
        self.groups = 3
        hidden = max(channels // 8, 4)
        # 全局统计 -> (N, 3*C) 的组权重 (迁移 S2-MLPv2: aW1 -> σ -> W2 -> softmax)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels * self.groups, kernel_size=1, bias=False),
        )

    def forward(self, feats):
        # feats: list of 3 tensors (N,C,T,H,W) -> stack (N,3,C,T,H,W)
        N, C, T, H, W = feats[0].shape
        x = torch.stack(feats, dim=1)                       # (N,3,C,T,H,W)
        feats_sum = x.sum(dim=1)                            # (N,C,T,H,W) (式2 a)
        pooled = feats_sum.mean(dim=(2, 3, 4), keepdim=True)  # GAP (N,C,1,1,1)
        attn = self.mlp(pooled)                              # (N,3C,1,1,1) (式3)
        attn = attn.view(N, self.groups, C, 1, 1, 1)
        attn = F.softmax(attn, dim=1)                        # 分组维 softmax (式4)
        out = (x * attn).sum(dim=1)                          # (N,C,T,H,W)
        return out


class TemporalAttention(nn.Module):
    """TA 时间注意力分支 (论文 Figure 5b)。

    强调帧间运动: 对 3 路移位特征计算相邻帧差分, 经时序卷积学习时间权重,
    softmax(时间维) 后加权求和 -> (N,C,T,H,W)。
    """

    def __init__(self, channels):
        super().__init__()
        self.groups = 3
        # 深度可分离时序卷积 (对 3*C 通道, 时间维建模)
        self.conv = nn.Conv3d(channels * self.groups, channels * self.groups,
                              kernel_size=(3, 1, 1), padding=(1, 0, 0),
                              groups=channels * self.groups, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        N, C, T, H, W = feats[0].shape
        x = torch.stack(feats, dim=1)                       # (N,3,C,T,H,W)
        # 相邻帧差分: diff(t) = x(t+1) - x(t), 首帧补 0 (强调运动)
        diff = torch.zeros_like(x)
        diff[:, :, :, 1:, :, :] = x[:, :, :, 1:, :, :] - x[:, :, :, :-1, :, :]
        x3c = x.reshape(N, self.groups * C, T, H, W)         # (N,3C,T,H,W)
        diff3c = diff.reshape(N, self.groups * C, T, H, W)
        # 时序卷积 + sigmoid -> 时间权重
        w = self.sigmoid(self.conv(diff3c))                 # (N,3C,T,H,W)
        w = w.reshape(N, self.groups, C, T, H, W)
        w = F.softmax(w, dim=3)                              # 沿 T 归一化
        out = (x * w).sum(dim=1)                             # (N,C,T,H,W)
        return out


class STSA(nn.Module):
    """Spatiotemporal Shift Attention Module (论文式 11-15)。

    Args:
        channels (int): 输入通道 (与输出一致, 即插即用, 残差保持).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # MLP1: C -> 3C (CM-MLP, 式11)
        self.mlp1 = nn.Conv3d(channels, channels * 3, kernel_size=1)
        # SA + TA 分支 (输入 3 路 C 通道, 输出 C)
        self.sa = SplitAttention(channels)
        self.ta = TemporalAttention(channels)
        # MLP2: 融合输出 (式15, 保持 C)
        self.mlp2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        input = x                                           # 残差用
        chi = self.mlp1(x)                                  # (N,3C,T,H,W) 式11
        # 式12 切三路
        C = self.channels
        x1 = chi[:, :C]
        x2 = chi[:, C:2 * C]
        x3 = chi[:, 2 * C:]
        # 式13/14 非对称空间移位 (χ3 不变)
        x1 = spatial_shift1(x1)
        x2 = spatial_shift2(x2)
        # 式15: SA(χ̃) + TA(χ̃), 再 MLP2
        out = self.sa([x1, x2, x3]) + self.ta([x1, x2, x3])
        out = self.mlp2(self.act(out))
        # 残差
        return input + out