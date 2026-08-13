import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPriorMixin:

    def _center_map(self, h, w, device):
        sigma = max(1.0, self.center_prior * max(h, w))
        ys = torch.arange(h, dtype=torch.float32, device=device).view(1, 1, 1, -1, 1)
        xs = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, 1, 1, -1)
        (cy, cx) = ((h - 1) / 2.0, (w - 1) / 2.0)
        g = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
        return g

    def _kp_prior_map(self, x, kp_map, h, w):
        if self.kp_prior <= 0 or kp_map is None:
            return None
        combined = kp_map.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        combined = combined.mean(dim=2, keepdim=True).squeeze(2)
        if combined.shape[-2] != h or combined.shape[-1] != w:
            combined = torch.nn.functional.interpolate(combined, size=(h, w), mode='bilinear', align_corners=False)
        return combined.unsqueeze(2)

    def _apply_mask(self, x, mask_map, h, w):
        if self.mask_prior <= 0:
            return None
        if mask_map is not None:
            if mask_map.shape[-2] != h or mask_map.shape[-1] != w:
                m4 = mask_map.squeeze(2)
                m4 = torch.nn.functional.interpolate(m4, size=(h, w), mode='nearest')
                mask_map = m4.unsqueeze(2)
            return mask_map
        return self._mouse_mask(x)

    def _mouse_mask(self, x, k=0.35):
        g = x.mean(dim=1)
        med = torch.median(g, dim=1, keepdim=True)[0]
        d = (g - med).abs().mean(dim=1, keepdim=True)
        th = d.mean(dim=(2, 3), keepdim=True) + k * (d.std(dim=(2, 3), keepdim=True) + 1e-06)
        return (d > th).float().detach().unsqueeze(2)

class MotionAttention(SpatialPriorMixin, nn.Module):

    def __init__(self, channels, reduction=16, diff_kernel=3, center_prior=0.0, mask_prior=0.0, kp_prior=0.0):
        super().__init__()
        self.kp_prior = float(kp_prior or 0.0)
        self.temporal_diff = nn.Conv3d(channels, channels, kernel_size=(diff_kernel, 1, 1), padding=(diff_kernel // 2, 0, 0), groups=channels, bias=False)
        with torch.no_grad():
            k = torch.zeros(1, 1, diff_kernel, 1, 1)
            center = diff_kernel // 2
            for i in range(diff_kernel):
                k[0, 0, i, 0, 0] = 2.0 if i == center else -1.0
            self.temporal_diff.weight.copy_(k)
        self.temporal_diff.skip_init = True
        self.temporal_diff.weight.requires_grad = True
        mid = max(channels // max(reduction, 1), 8)
        self.ch_gate = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Conv3d(channels, mid, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(mid, channels, kernel_size=1, bias=False))
        self.sp_gate = nn.Conv3d(channels, 1, kernel_size=(1, 1, 1), bias=False)
        self.center_prior = float(center_prior or 0.0)
        self.mask_prior = float(mask_prior or 0.0)
        self._capture = False
        self._capture_cache = None

    def forward(self, x, kp_map=None, mask_map=None):
        (B, C, T, H, W) = x.shape
        diff = torch.abs(self.temporal_diff(x))
        w_c = torch.sigmoid(self.ch_gate(diff))
        w_s = torch.sigmoid(self.sp_gate(diff))
        mask = None
        if self.center_prior > 0:
            w_p = 0.5 + 0.5 * self._center_map(H, W, x.device)
            w_s = w_s * w_p
        mask = self._apply_mask(x, mask_map, H, W)
        if mask is not None:
            w_s = w_s * (1.0 - self.mask_prior + self.mask_prior * mask)
        kp = self._kp_prior_map(x, kp_map, H, W)
        if kp is not None:
            w_s = w_s * (1.0 - self.kp_prior + self.kp_prior * kp)
        out = x * w_c * w_s
        if getattr(self, '_capture', False):
            self._capture_cache = (diff, w_c, w_s, mask)
        return out

class CBAMChannelGate(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv3d(channels, channels // max(reduction, 1), kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(channels // max(reduction, 1), channels, kernel_size=1, bias=False))

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        mx = torch.max(x, dim=2, keepdim=True)[0]
        mx = torch.max(mx, dim=3, keepdim=True)[0]
        mx = torch.max(mx, dim=4, keepdim=True)[0]
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * w

class CBAMSpatialGate(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size // 2, kernel_size // 2), bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        (mx, _) = torch.max(x, dim=1, keepdim=True)
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
        (B, C, T, H, W) = x.shape
        xt = x.mean(dim=2, keepdim=True)
        x_h = self.pool_h(xt)
        x_w = self.pool_w(xt).permute(0, 1, 2, 4, 3)
        y = torch.cat([x_h, x_w], dim=3)
        y = self.act(self.bn1(self.conv1(y)))
        (x_h, x_w) = torch.split(y, [H, W], dim=3)
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
        self.ch_gate = nn.Sequential(nn.Conv3d(channels, mid, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(mid, channels, kernel_size=1, bias=False))

    @torch.no_grad()
    def _split_maps(self, x, E):
        (B, C, T, H, W) = x.shape
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
        body = x.mean(dim=2).mean(dim=1, keepdim=True)
        th = body.view(B, -1).mean(dim=1, keepdim=True).view(B, 1, 1, 1)
        mask = (body > th).float()
        if mask.mean() < 0.2:
            mask = torch.ones_like(mask)
        wsum = mask.sum(dim=(2, 3), keepdim=True) + 1e-06
        cy = (mask * ys).sum(dim=(2, 3), keepdim=True) / wsum
        cx = (mask * xs).sum(dim=(2, 3), keepdim=True) / wsum
        (dy, dx) = (ys - cy, xs - cx)
        m_yy = (mask * dy * dy).sum(dim=(2, 3), keepdim=True) / wsum
        m_xx = (mask * dx * dx).sum(dim=(2, 3), keepdim=True) / wsum
        m_xy = (mask * dy * dx).sum(dim=(2, 3), keepdim=True) / wsum
        th_ax = 0.5 * torch.atan2(2 * m_xy, m_xx - m_yy)
        (vx, vy) = (torch.cos(th_ax), torch.sin(th_ax))
        wm = E.sum(dim=(1, 2, 3), keepdim=True) + 1e-06
        my = (E * ys).sum(dim=(1, 2, 3), keepdim=True) / wm
        mx = (E * xs).sum(dim=(1, 2, 3), keepdim=True) / wm
        drift = (mx - cx) * vx + (my - cy) * vy
        sign = torch.where(drift >= 0, torch.tensor(1.0, device=x.device), torch.tensor(-1.0, device=x.device))
        (vx, vy) = (vx * sign, vy * sign)
        u = (xs - cx) * vx + (ys - cy) * vy
        front = torch.sigmoid(u / (self.soft_sigma + 1e-06)).clamp(0.02, 0.98)
        back = 1.0 - front
        return (front.view(B, 1, 1, H, W), back.view(B, 1, 1, H, W))

    def forward(self, x):
        (B, C, T, H, W) = x.shape
        if T > 1:
            diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
            diff = F.pad(diff, (0, 0, 0, 0, 0, 1))
        else:
            diff = torch.zeros_like(x)
        E = diff.mean(dim=1)
        (front, back) = self._split_maps(x, E)
        s_front = (E.unsqueeze(1) * front).sum(dim=(2, 3, 4), keepdim=True) / front.sum(dim=(2, 3, 4), keepdim=True)
        s_back = (E.unsqueeze(1) * back).sum(dim=(2, 3, 4), keepdim=True) / back.sum(dim=(2, 3, 4), keepdim=True)
        w_c = torch.sigmoid(self.ch_gate(diff))
        w_s = front * s_front + back * s_back
        w_s = w_s / (w_s.mean(dim=(3, 4), keepdim=True) + 1e-06)
        out = x * w_c * (0.5 + 0.5 * w_s)
        if getattr(self, '_capture', False):
            self._capture_cache = (front, back, s_front, s_back, E)
        return out

class MotionStatAttention(SpatialPriorMixin, nn.Module):

    def __init__(self, channels, reduction=16, center_prior=0.0, mask_prior=0.0, kp_prior=0.0):
        super().__init__()
        self.kp_prior = float(kp_prior or 0.0)
        self.channels = channels
        mid = max(channels // max(reduction, 1), 8)
        self.sp_gate = nn.Conv3d(4, 1, kernel_size=1, bias=True)
        self.ch_gate = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Conv3d(channels, mid, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(mid, channels, kernel_size=1, bias=False))
        self.center_prior = float(center_prior or 0.0)
        self.mask_prior = float(mask_prior or 0.0)
        self._capture = False
        self._capture_cache = None

    def _fast_ratio(self, x):
        (B, C, T, H, W) = x.shape
        if T > 4:
            m1 = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean(dim=1)
            m1 = F.pad(m1, (0, 0, 0, 0, 0, 1))
            m4 = torch.abs(x[:, :, 4:] - x[:, :, :-4]).mean(dim=1)
            m4 = F.pad(m4, (0, 0, 0, 0, 0, 4))
        else:
            m1 = torch.zeros(B, T, H, W, device=x.device, dtype=x.dtype)
            m4 = torch.zeros(B, T, H, W, device=x.device, dtype=x.dtype)
        r = m1 / (m4 + 0.0001)
        return (m1, m4, r)

    def forward(self, x, kp_map=None, mask_map=None):
        (B, C, T, H, W) = x.shape
        (m1, m4, r) = self._fast_ratio(x)
        mu = m1.mean(dim=1, keepdim=True)
        sg = torch.std(m1, dim=1, keepdim=True, unbiased=False) + 1e-06
        r_n = torch.sigmoid((r - 1.0) * 2.0)
        e = mu
        c = e.sum(dim=(2, 3), keepdim=True) ** 2 / (e.pow(2).sum(dim=(2, 3), keepdim=True) + 1e-08)
        c = c / max(float(H * W), 1)
        stats = torch.stack([r_n, mu.expand(-1, T, -1, -1), sg.expand(-1, T, -1, -1), c.expand(B, T, H, W)], dim=1)
        w_s = torch.sigmoid(self.sp_gate(stats))
        mask = None
        if self.center_prior > 0:
            w_p = 0.5 + 0.5 * self._center_map(H, W, x.device)
            w_s = w_s * w_p
        mask = self._apply_mask(x, mask_map, H, W)
        if mask is not None:
            w_s = w_s * (1.0 - self.mask_prior + self.mask_prior * mask)
        kp = self._kp_prior_map(x, kp_map, H, W)
        if kp is not None:
            w_s = w_s * (1.0 - self.kp_prior + self.kp_prior * kp)
        w_c = torch.sigmoid(self.ch_gate(x))
        out = x * w_c * w_s
        if getattr(self, '_capture', False):
            self._capture_cache = (r_n, mu, sg, c, w_s, m1, mask)
        return out

def build_attention(name, channels, reduction=None, num_keypoints=27, center_prior=0.0, mask_prior=0.0, kp_prior=0.0):
    name = (name or 'none').lower()
    if name in ('motion', 'motion+part', 'part_motion'):
        base = MotionAttention(channels, reduction=reduction if reduction is not None else 16, center_prior=center_prior, mask_prior=mask_prior, kp_prior=kp_prior)
        if name in ('motion+part', 'part_motion'):
            return nn.Sequential(base, PartMotionAttention(channels, reduction=reduction if reduction is not None else 16))
        return base
    if name in ('motion_stat', 'motion-stat', 'point'):
        return MotionStatAttention(channels, reduction=reduction if reduction is not None else 16, center_prior=center_prior, mask_prior=mask_prior, kp_prior=kp_prior)
    if name == 'part':
        return PartMotionAttention(channels, reduction=reduction if reduction is not None else 16)
    if name == 'cbam':
        return CBAM(channels, reduction=reduction if reduction is not None else 16)
    if name == 'coordinate':
        return CoordinateAttention(channels, reduction=reduction if reduction is not None else 32)
    if name == 'pose_prompt':
        return PosePromptAttention(channels, num_keypoints=num_keypoints)
    if name == 'none':
        return nn.Identity()
    raise ValueError(f'unknown attention: {name}')
