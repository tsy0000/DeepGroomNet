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
    """SA 空间注意力分支 (论文 Figure 5a).

    严格按图5a 流程:
      input (N,K,T,H,W,C) -> Reshape (N,K,M,C)  M=T*H*W
      s = Sum_M(input)  (N,K,C)                  每分组时空统计
      att = MLP2(GELU(MLP1(s)))  (N,K,C)         沿 C 维的共享 MLP
      alpha = softmax(att, dim=K)  (N,K,1,C)     K 分组维 softmax
      out = Sum_K(input ⊙ alpha) -> (N,T,H,W,C)
    易错点: softmax 在 K(分组)维, 不在 C 维.
    """

    def __init__(self, channels):
        super().__init__()
        self.groups = 3
        # MLP2(GELU(MLP1)): 作用在 C 维, 所有 (n,k) 共享
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, feats):
        # feats: list of 3 tensors (N,C,T,H,W)  (即 K=3 分组)
        N, C, T, H, W = feats[0].shape
        x = torch.stack(feats, dim=1)              # (N,K,C,T,H,W)  K=3
        s = x.sum(dim=(3, 4, 5))                   # 每分组 Sum_M -> (N,K,C)
        att = self.mlp(s)                          # MLP2(GELU(MLP1)) -> (N,K,C)
        alpha = F.softmax(att, dim=1)              # K 维 softmax (N,K,C)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (N,K,C,1,1,1)
        out = (x * alpha).sum(dim=1)               # Sum_K 加权 -> (N,C,T,H,W)
        return out


class TemporalAttention(nn.Module):
    """TA 时间注意力分支 (论文 Figure 5b).

    严格按图5b 流程:
      input (N,K,T,H,W,C) -> Permute+Sum_K -> (N,C,T,H,W)   (丢弃分组维)
      Pad(Diff(x))                                  帧间差分 + 时间 pad
      mask = Sigmoid(Conv3d(diff))  (N,C,T,H,W)     时序注意力掩码
      out = x ⊙ mask -> (N,T,H,W,C)
    易错点: TA 用 sigmoid 掩码, 没有 softmax! (softmax 只在 SA 的 K 维)
    """

    def __init__(self, channels):
        super().__init__()
        # 3D 时序卷积 (仅时间维核=3), 从帧差生成掩码
        self.conv = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                              padding=(1, 0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        N, C, T, H, W = feats[0].shape
        x = torch.stack(feats, dim=1).sum(dim=1)     # Sum_K -> (N,C,T,H,W)
        # 帧间差分 + 首帧补0 (保持时间维 T)
        diff = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0))
        mask = self.sigmoid(self.conv(diff))         # (N,C,T,H,W) sigmoid 掩码
        return x * mask


class STSA(nn.Module):
    """Spatiotemporal Shift Attention Module (论文式 11-15)。

    MLP1 / MLP2 采用论文引用的 MLP-mixer[23]/ResMLP[24] 结构:
      Linear -> GELU -> Linear (两层通道混合 MLP, 用 1x1 Conv3d 实现).
    图3 存在残差 shortcut (stage 输入直连末尾 ⊕), 故返回 input + out.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        C = channels
        # MLP1: C -> 3C (两层: Linear->GELU->Linear, 式11)
        self.mlp1 = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(C, C * 3, kernel_size=1),
        )
        # SA + TA 分支 (输入 3 路 C 通道, 输出 C)
        self.sa = SplitAttention(channels)
        self.ta = TemporalAttention(channels)
        # MLP2: C -> C 融合 (两层: Linear->GELU->Linear, 式15)
        self.mlp2 = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(C, C, kernel_size=1),
        )

    def forward(self, x):
        input = x                                           # 残差用 (图3 shortcut)
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
        out = self.mlp2(out)
        # 残差 (图3: 输入直连末尾 ⊕)
        return input + out