"""
模块A（Solution 1）：把注意力引到小鼠身上的注意力模块（文献证据链选择）。

提供三个即插即用注意力层（均有公开代码/文献）：
  - MotionAttention    : 运动感知注意力（默认）—— 行为 = 运动模式（FaceGroom 脸搓 vs BodyGroom 身体舔
                         的差异主要在运动模式），时序差分 → 运动显著区域/通道增强。
                         依据：TEA (CVPR 2020) 时序激励；BehaveAI (PLOS Biology 2026)
                         "color-from-motion" 证明运动信息编码显著提升动物行为分类。
  - CBAMChannelGate/SpatialGate : Woo et al., "CBAM", ECCV 2018
  - CoordinateAttention : Hou et al., "Coordinate Attention", CVPR 2021（位置先验，部位定位）

用法：
  from models.attention import build_attention
  attn = build_attention("motion", channels=192)
  out = attn(x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPriorMixin:
    """空间先验共享实现：中心高斯 + 中值背景减除小鼠掩码（确定性，零标注）。"""

    def _center_map(self, h, w, device):
        """归一化中心高斯：峰值 1（画面中心），边缘 ~0。确定性，无学习参数。"""
        sigma = max(1.0, self.center_prior * max(h, w))
        ys = torch.arange(h, dtype=torch.float32, device=device).view(1, 1, 1, -1, 1)
        xs = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, 1, 1, -1)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        g = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
        return g

    def _kp_prior_map(self, x, kp_map, h, w):
        """关键点空间先验（PoseC3D 伪热图）：kp_map [B,C,T,H0,W0]（关键点+肢体通道）
        → 合并为 [B,1,H,W]（通道 max 池化后空间聚合），门乘 (1-kp_prior)+kp_prior·map。
        热图只编码关键点几何 → 对光照/背景/颜色分布鲁棒（新视频泛化）。
        """
        if self.kp_prior <= 0 or kp_map is None:
            return None
        combined = kp_map.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        combined = combined.mean(dim=2, keepdim=True).squeeze(2)
        if combined.shape[-2] != h or combined.shape[-1] != w:
            combined = torch.nn.functional.interpolate(
                combined, size=(h, w), mode="bilinear", align_corners=False)
        return combined.unsqueeze(2)

    def _apply_mask(self, x, mask_map, h, w):
        """掩码先验：优先用输入级掩码（像素计算，真实小鼠位置）；无则退回特征级。"""
        if self.mask_prior <= 0:
            return None
        if mask_map is not None:
            if mask_map.shape[-2] != h or mask_map.shape[-1] != w:
                m4 = mask_map.squeeze(2)
                m4 = torch.nn.functional.interpolate(m4, size=(h, w), mode="nearest")
                mask_map = m4.unsqueeze(2)
            return mask_map
        return self._mouse_mask(x)

    def _mouse_mask(self, x, k=0.35):
        """中值背景减除掩码（Jhuang et al. VAIB'10 思路，特征域实现）。

        x: [B,C,T,H,W] -> mask [B,1,1,H,W]（0=背景，1=小鼠）
        通道均值 -> 时维中值背景 -> 前景响应 |x-med| -> 均值+k·std 阈值。
        detach()：确定性先验，不参与梯度。
        """
        g = x.mean(dim=1)
        med = torch.median(g, dim=1, keepdim=True)[0]
        d = (g - med).abs().mean(dim=1, keepdim=True)
        th = d.mean(dim=(2, 3), keepdim=True) + k * (d.std(dim=(2, 3), keepdim=True) + 1e-6)
        return (d > th).float().detach().unsqueeze(2)


class MotionAttention(SpatialPriorMixin, nn.Module):
    """运动感知注意力：把特征中的"运动区域"和"运动通道"增强。

    行为识别中时序差分/运动信息对行为分类的增益方向明确（行为=运动模式），
    且对稀有行为（FaceGroom/BodyGroom 帧占比 <5%）提供"运动显著"先验。

    center_prior: >0 时叠加中心高斯空间先验（YOLO 裁剪后小鼠必然居中，
                  该确定性结构先验把注意力聚焦小鼠、抑制边缘背景噪声）。

    forward: x[B,C,T,H,W] -> 同 shape
      w_c: 通道级运动强度（时序差分能量，SE 式激励）-> 通道门控
      w_s: 空间运动显著图（跨通道差分能量）-> 空间门控
      w_p(可选): 中心高斯先验 -> 空间门控
    """
    def __init__(self, channels, reduction=16, diff_kernel=3, center_prior=0.0,
                 mask_prior=0.0, kp_prior=0.0):
        super().__init__()
        self.kp_prior = float(kp_prior or 0.0)
        self.temporal_diff = nn.Conv3d(
            channels, channels, kernel_size=(diff_kernel, 1, 1),
            padding=(diff_kernel // 2, 0, 0), groups=channels, bias=False)
        with torch.no_grad():
            k = torch.zeros(1, 1, diff_kernel, 1, 1)
            center = diff_kernel // 2
            for i in range(diff_kernel):
                k[0, 0, i, 0, 0] = 2.0 if i == center else -1.0
            self.temporal_diff.weight.copy_(k)
        self.temporal_diff.skip_init = True
        self.temporal_diff.weight.requires_grad = True

        mid = max(channels // max(reduction, 1), 8)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=False),
        )
        self.sp_gate = nn.Conv3d(channels, 1, kernel_size=(1, 1, 1), bias=False)
        self.center_prior = float(center_prior or 0.0)
        self.mask_prior = float(mask_prior or 0.0)
        self._capture = False
        self._capture_cache = None

    def forward(self, x, kp_map=None, mask_map=None):
        dtype = x.dtype
        x = x.float()
        if kp_map is not None:
            kp_map = kp_map.float()
        if mask_map is not None:
            mask_map = mask_map.float()
        with torch.cuda.amp.autocast(enabled=False):
            B, C, T, H, W = x.shape
            diff = torch.abs(self.temporal_diff(x))
            w_c = torch.sigmoid(self.ch_gate(diff))
            w_s = torch.sigmoid(self.sp_gate(diff))
            mask = None
            if self.center_prior > 0:
                w_p = 0.5 + 0.5 * self._center_map(H, W, x.device)
                w_s = w_s * w_p
            mask = self._apply_mask(x, mask_map, H, W)
            if mask is not None:
                w_s = w_s * ((1.0 - self.mask_prior) + self.mask_prior * mask)
            kp = self._kp_prior_map(x, kp_map, H, W)
            if kp is not None:
                w_s = w_s * ((1.0 - self.kp_prior) + self.kp_prior * kp)
            out = x * w_c * w_s
        if getattr(self, "_capture", False):
            self._capture_cache = (diff, w_c, w_s, mask)
        return out.to(dtype)


class CBAMChannelGate(nn.Module):
    """CBAM channel attention（支持时空特征 [B,C,T,H,W]，对 T 求均值后做通道注意）"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // max(reduction, 1), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // max(reduction, 1), channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        mx = torch.max(x, dim=2, keepdim=True)[0]
        mx = torch.max(mx, dim=3, keepdim=True)[0]
        mx = torch.max(mx, dim=4, keepdim=True)[0]
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w


class CBAMSpatialGate(nn.Module):
    """CBAM spatial attention：对通道维取均值/最大值 -> 1 通道空间权重（跨 T 共享）"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size),
                              padding=(0, kernel_size // 2, kernel_size // 2), bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        fused = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(fused))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = CBAMChannelGate(channels, reduction)
        self.spatial = CBAMSpatialGate(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


class CoordinateAttention(nn.Module):
    """Coordinate Attention（3D 视频适配版）。
    官方：Hou et al., CVPR 2021, https://github.com/Andrew-Qibin/CoordAttention
    这里对 T 维做平均，把 H/W 两个方向的位置信息编码成注意力权重。
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        mid = max(channels // reduction, 8)
        self.conv1 = nn.Conv3d(channels, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv3d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv3d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T, H, W = x.shape
        xt = x.mean(dim=2, keepdim=True)

        x_h = self.pool_h(xt)
        x_w = self.pool_w(xt).permute(0, 1, 2, 4, 3)

        y = torch.cat([x_h, x_w], dim=3)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=3)
        x_w = x_w.permute(0, 1, 2, 4, 3)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class PosePromptAttention(nn.Module):
    def __init__(self, channels, num_keypoints=27, sigma=2.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        self.gate = nn.Conv3d(num_keypoints, channels, kernel_size=1, bias=True)

    def forward(self, x, keypoint_map):
        """
        x            : [B, C, T, H, W]
        keypoint_map : [B, K, T, H, W]  关键点高斯热图（每个关键点在每帧一张热图）
        """
        g = torch.sigmoid(self.gate(keypoint_map))
        return x * g


class PartMotionAttention(nn.Module):
    def __init__(self, channels, reduction=16, front_ratio=0.4, soft_sigma=0.2):
        super().__init__()
        self.channels = channels
        self.front_ratio = front_ratio
        self.soft_sigma = soft_sigma
        self._capture = False
        self._capture_cache = None
        mid = max(channels // max(reduction, 1), 8)
        self.ch_gate = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=False),
        )

    @torch.no_grad()
    def _split_maps(self, x, E):
        """沿小鼠身体主轴把空间分裂为 front/back 两张部位图。

        x: [B,C,T,H,W], E: [B,T,H,W]（运动能量，>=0）
        主轴来源 = 静态身体掩码（时均帧强度阈值化）的 PCA——小鼠身体为长条形，
        长轴 = 头尾轴（运动团块本身近各向同性，不可用）。头侧 = 运动质心相对
        身体中心沿主轴的分量符号。返回 (front, back): [B,1,1,H,W]，和为 1。
        """
        B, C, T, H, W = x.shape
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
        body = x.mean(dim=2).mean(dim=1, keepdim=True)
        th = body.view(B, -1).mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        mask = (body > th).float()
        if mask.mean() < 0.2:
            mask = torch.ones_like(mask)
        wsum = mask.sum(dim=(2, 3), keepdim=True) + 1e-6
        cy = (mask * ys).sum(dim=(2, 3), keepdim=True) / wsum
        cx = (mask * xs).sum(dim=(2, 3), keepdim=True) / wsum
        dy, dx = ys - cy, xs - cx
        m_yy = (mask * dy * dy).sum(dim=(2, 3), keepdim=True) / wsum
        m_xx = (mask * dx * dx).sum(dim=(2, 3), keepdim=True) / wsum
        m_xy = (mask * dy * dx).sum(dim=(2, 3), keepdim=True) / wsum
        th_ax = 0.5 * torch.atan2(2 * m_xy, m_xx - m_yy)
        vx, vy = torch.cos(th_ax), torch.sin(th_ax)
        wm = E.sum(dim=(1, 2, 3), keepdim=True) + 1e-6
        my = (E * ys).sum(dim=(1, 2, 3), keepdim=True) / wm
        mx = (E * xs).sum(dim=(1, 2, 3), keepdim=True) / wm
        drift = (mx - cx) * vx + (my - cy) * vy
        sign = torch.where(drift >= 0, torch.tensor(1.0, device=x.device),
                           torch.tensor(-1.0, device=x.device))
        vx, vy = vx * sign, vy * sign
        u = (xs - cx) * vx + (ys - cy) * vy
        front = torch.sigmoid(u / (self.soft_sigma + 1e-6)).clamp(0.02, 0.98)
        back = 1.0 - front
        return front.view(B, 1, 1, H, W), back.view(B, 1, 1, H, W)

    def forward(self, x):
        B, C, T, H, W = x.shape
        if T > 1:
            diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            diff = F.pad(diff, (0, 0, 0, 0, 0, 1))
        else:
            diff = torch.zeros_like(x)
        E = diff.mean(dim=1)
        front, back = self._split_maps(x, E)
        s_front = (E.unsqueeze(1) * front).sum(dim=(2, 3, 4), keepdim=True) / \
                  front.sum(dim=(2, 3, 4), keepdim=True)
        s_back = (E.unsqueeze(1) * back).sum(dim=(2, 3, 4), keepdim=True) / \
                 back.sum(dim=(2, 3, 4), keepdim=True)
        w_c = torch.sigmoid(self.ch_gate(diff))
        w_s = (front * s_front + back * s_back)
        w_s = w_s / (w_s.mean(dim=(3, 4), keepdim=True) + 1e-6)
        out = x * w_c * (0.5 + 0.5 * w_s)
        if getattr(self, "_capture", False):
            self._capture_cache = (front, back, s_front, s_back, E)
        return out


class MotionStatAttention(SpatialPriorMixin, nn.Module):
    def __init__(self, channels, reduction=16, center_prior=0.0, mask_prior=0.0,
                 kp_prior=0.0):
        super().__init__()
        self.kp_prior = float(kp_prior or 0.0)
        self.channels = channels
        mid = max(channels // max(reduction, 1), 8)
        self.sp_gate = nn.Conv3d(4, 1, kernel_size=1, bias=True)
        self.ch_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, kernel_size=1, bias=False),
        )
        self.center_prior = float(center_prior or 0.0)
        self.mask_prior = float(mask_prior or 0.0)
        self._capture = False
        self._capture_cache = None

    def _fast_ratio(self, x):
        """返回 (m1, m4, r)：时滞 1/4 差分与快速振荡比 r=m1/(m4+ε)（BehaveAI 式）。

        注意：差分在"通道绝对差分"上进行，不能用通道均值后差分——
        官方 LayerNorm(channels_first) 会把逐位置通道均值归零（weight=1, bias=0），
        且 block 初始化 LayerScale=1e-6 近恒等，均值路径的时序差分坍缩为 0
        （实测 1e-8 级）→ 运动统计全灭、掩码全零、注意力把特征清零。
        |Δx| 先按通道取绝对再平均保留幅度，任意通道的时序变化都能被捕获。
        """
        B, C, T, H, W = x.shape
        if T > 4:
            m1 = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean(dim=1)
            m1 = F.pad(m1, (0, 0, 0, 0, 0, 1))
            m4 = torch.abs(x[:, :, 4:] - x[:, :, :-4]).mean(dim=1)
            m4 = F.pad(m4, (0, 0, 0, 0, 0, 4))
        else:
            m1 = torch.zeros(B, T, H, W, device=x.device, dtype=x.dtype)
            m4 = torch.zeros(B, T, H, W, device=x.device, dtype=x.dtype)
        r = m1 / (m4 + 1e-4)
        return m1, m4, r

    def forward(self, x, kp_map=None, mask_map=None):
        dtype = x.dtype
        x = x.float()
        if kp_map is not None:
            kp_map = kp_map.float()
        if mask_map is not None:
            mask_map = mask_map.float()
        with torch.cuda.amp.autocast(enabled=False):
            B, C, T, H, W = x.shape
            m1, m4, r = self._fast_ratio(x)
            mu = m1.mean(dim=1, keepdim=True)
            sg = torch.std(m1, dim=1, keepdim=True, unbiased=False) + 1e-6
            r_n = torch.sigmoid((r - 1.0) * 2.0)
            e = mu
            c = (e.sum(dim=(2, 3), keepdim=True) ** 2) / (e.pow(2).sum(dim=(2, 3), keepdim=True) + 1e-8)
            c = c / max(float(H * W), 1)
            stats = torch.stack([r_n, mu.expand(-1, T, -1, -1),
                                 sg.expand(-1, T, -1, -1),
                                 c.expand(B, T, H, W)], dim=1)
            w_s = torch.sigmoid(self.sp_gate(stats))
            mask = None
            if self.center_prior > 0:
                w_p = 0.5 + 0.5 * self._center_map(H, W, x.device)
                w_s = w_s * w_p
            mask = self._apply_mask(x, mask_map, H, W)
            if mask is not None:
                w_s = w_s * ((1.0 - self.mask_prior) + self.mask_prior * mask)
            kp = self._kp_prior_map(x, kp_map, H, W)
            if kp is not None:
                w_s = w_s * ((1.0 - self.kp_prior) + self.kp_prior * kp)
            w_c = torch.sigmoid(self.ch_gate(x))
            out = x * w_c * w_s
        if getattr(self, "_capture", False):
            self._capture_cache = (r_n, mu, sg, c, w_s, m1, mask)
        return out.to(dtype)


def build_attention(name, channels, reduction=None, num_keypoints=27, center_prior=0.0,
                    mask_prior=0.0, kp_prior=0.0):
    name = (name or "none").lower()
    if name in ("motion", "motion+part", "part_motion"):
        base = MotionAttention(channels,
                               reduction=reduction if reduction is not None else 16,
                               center_prior=center_prior, mask_prior=mask_prior,
                               kp_prior=kp_prior)
        if name in ("motion+part", "part_motion"):
            return nn.Sequential(base,
                                 PartMotionAttention(channels,
                                                     reduction=reduction if reduction is not None else 16))
        return base
    if name in ("motion_stat", "motion-stat", "point"):
        return MotionStatAttention(channels,
                                   reduction=reduction if reduction is not None else 16,
                                   center_prior=center_prior, mask_prior=mask_prior,
                                   kp_prior=kp_prior)
    if name == "part":
        return PartMotionAttention(channels,
                                   reduction=reduction if reduction is not None else 16)
    if name == "cbam":
        return CBAM(channels, reduction=reduction if reduction is not None else 16)
    if name == "coordinate":
        return CoordinateAttention(channels,
                                   reduction=reduction if reduction is not None else 32)
    if name == "pose_prompt":
        return PosePromptAttention(channels, num_keypoints=num_keypoints)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"unknown attention: {name}")
