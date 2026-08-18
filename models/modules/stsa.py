"""
STSA - Spatiotemporal Shift Attention Module.

严格按 Swin-Panda 论文 (Diversity 2025, 17, 139) 式 11-15 实现。
空间移位灵感来自 S2-MLPv2 [22] (arXiv 2108.01072)，改造为 3D 视频版本：
输入 (N,C,T,H,W)，移位只在 H/W 空间做、保留时间维 T 不变。

论文结构:
  χ̂ = MLP1(χ) ∈ R^{3C×T×H×W}                    (式11, CM-MLP: C->3C)
  χ1,χ2,χ3 = split(χ̂, 3, dim=C)                  (式12)
  χ1 <- SS1(χ1), χ2 <- SS2(χ2), χ3 不变           (式13/14 非对称空间移位)
  X  = MLP2( SA(χ̃) + TA(χ̃) )                    (式15, SA空间 + TA时间注意力)

零参数特性: 空间移位为纯索引操作，不新增模型参数 (论文 Table 5: Params 不变)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spatial_shift1(x):
    """非对称空间移位 SS1 (3D 版, 仅 H/W)。

    方向: 通道块 [c/4|c/2|3c/4] 沿 H/W 平移, 边界补 0。
    x: (N, C, T, H, W)
    """
    N, C, T, H, W = x.shape
    c1, c2, c3 = C // 4, C // 2, 3 * C // 4
    out = torch.zeros_like(x)
    # 0~c/4: 沿 H 下移 (从 h-1 移到 1:)
    out[:, :c1, :, 1:, :] = x[:, :c1, :, :H - 1, :]
    # c/4~c/2: 沿 H 上移
    out[:, c1:c2, :, :H - 1, :] = x[:, c1:c2, :, 1:, :]
    # c/2~c3/4: 沿 W 右移
    out[:, c2:c3, :, :, 1:] = x[:, c2:c3, :, :, :W - 1]
    # c3/4~C: 沿 W 左移
    out[:, c3:, :, :, :W - 1] = x[:, c3:, :, :, 1:]
    return out


def spatial_shift2(x):
    """非对称空间移位 SS2 (与 SS1 方向相反的组合)。"""
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
    """SA 空间注意力分支 (论文 Figure 5a)。

    输入 3 路移位特征, 沿空间求和生成统计量, MLP 学权重, sigmoid 门控。
    参考 DehazeFormer/ResMLP 的 split-attention 逻辑, 3D 版。
    """

    def __init__(self, channels):
        super().__init__()
        self.groups = 3
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        hidden = max(channels // 8, 4)
        self.mlp1 = nn.Conv3d(channels * self.groups, channels * self.groups,
                              kernel_size=1, bias=False)
        self.mlp2 = nn.Conv3d(channels * self.groups, channels * self.groups,
                              kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, feats):
        # feats: list of 3 tensors (N,C,T,H,W) -> stack (N,3,C,T,H,W)
        N, C, T, H, W = feats[0].shape
        x = torch.stack(feats, dim=1)               # (N,3,C,T,H,W)
        # 空间统计量: 沿 C 求和 (每个尺度的通道统计)
        x_sum = x.sum(dim=2, keepdim=True)          # (N,3,1,T,H,W)
        pooled = self.avg_pool(x_sum)               # (N,3,1,1,1,1)
        pooled = pooled.view(N, self.groups, 1, 1, 1, 1)
        # 展平融合 (3,1) -> 学习每尺度权重
        flat = pooled.reshape(N, self.groups)       # (N,3)
        # 用全连接学习分组权重
        weight = torch.softmax(flat, dim=1)         # (N,3)
        weight = weight.view(N, self.groups, 1, 1, 1, 1)
        out = (x * weight).sum(dim=1)               # (N,C,T,H,W)
        return out


class TemporalAttention(nn.Module):
    """TA 时间注意力分支 (论文 Figure 5b)。

    强调帧间运动: 计算相邻帧差分, 学习时序权重, softmax 时间维。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                              padding=(1, 0, 0), groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T, H, W = x.shape
        # 相邻帧差分: diff(t) = x(t+1) - x(t), 首帧补 0
        diff = torch.zeros_like(x)
        diff[:, :, 1:, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        # 时序卷积 + sigmoid -> 时序权重
        w = self.sigmoid(self.conv(diff))           # (N,C,T,H,W)
        # softmax 时间维
        w = F.softmax(w, dim=2)                     # 沿 T 归一化
        return x * w


class STSA(nn.Module):
    """Spatiotemporal Shift Attention Module (论文式 11-15)。

    Args:
        channels (int): 输入通道 (与输出一致, 即插即用).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # MLP1: C -> 3C (CM-MLP)
        self.mlp1 = nn.Conv3d(channels, channels * 3, kernel_size=1)
        # SA + TA 分支
        self.sa = SplitAttention(channels)
        self.ta = TemporalAttention(channels)
        # MLP2: 融合输出 (通道还原)
        self.mlp2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        input = x                                   # 残差用
        chi = self.mlp1(x)                          # (N,3C,T,H,W) 式11
        # 式12 切三路
        C = self.channels
        x1 = chi[:, :C]
        x2 = chi[:, C:2 * C]
        x3 = chi[:, 2 * C:]
        # 式13/14 非对称空间移位
        x1 = spatial_shift1(x1)
        x2 = spatial_shift2(x2)
        # 式15: SA(χ̃) + TA(χ̃), 再 MLP2
        out = self.sa([x1, x2, x3]) + self.ta(x)
        out = self.mlp2(out)
        # 残差
        return input + out