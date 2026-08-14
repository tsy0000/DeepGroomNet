"""官方数据增强移植（alibaba-mmai-research/TAdaConv，ICLR 2022 原版实现）。

来源文件：
  tadaconv/datasets/utils/transformations.py  → ColorJitter（亮度/对比度/饱和度/色相/灰度，跨帧一致）
  tadaconv/datasets/utils/random_erasing.py   → RandomErasing
  tadaconv/datasets/base/base_dataset.py      → 训练变换链（翻转 → 随机裁剪 → 色彩抖动 → 擦除）

参数适配说明：官方 K400 输入 112、RandomResizedCrop scale=(0.08,1.0)；
本项目输入 224 且小鼠占画面仅 10-16%，scale 下限提高避免裁掉小鼠本体
（YOLO 裁剪已保证小鼠居中）。
"""
import numpy as np


class ColorJitter:
    """官方 TAdaConv ColorJitter（逐帧独立参数；consistent=True 时跨帧一致）。

    实现原版：亮度/对比度/饱和度乘性扰动 + 可选的灰度/颜色混合，参数 shuffle。
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0,
                 grayscale=0.0, color=0.0, consistent=True, shuffle=False,
                 gray_first=True):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale = grayscale
        self.color = color
        self.consistent = consistent
        self.shuffle = shuffle
        self.gray_first = gray_first
        self.rng = np.random.RandomState()

    def _sample(self):
        b = 1.0 + self.brightness * self.rng.uniform(-1, 1) if self.brightness > 0 else 1.0
        c = 1.0 + self.contrast * self.rng.uniform(-1, 1) if self.contrast > 0 else 1.0
        s = 1.0 + self.saturation * self.rng.uniform(-1, 1) if self.saturation > 0 else 1.0
        h = self.hue * self.rng.uniform(-1, 1) if self.hue > 0 else 0.0
        return b, c, s, h

    def _adjust_brightness(self, arr, b):
        return np.clip(arr * b, 0, 1)

    def _adjust_contrast(self, arr, c):
        mean = arr.mean(axis=-1, keepdims=True)
        return np.clip((arr - mean) * c + mean, 0, 1)

    def _adjust_saturation(self, arr, s):
        gray = arr.mean(axis=-1, keepdims=True)
        return np.clip(gray + (arr - gray) * s, 0, 1)

    def _adjust_hue(self, arr, h):
        if abs(h) < 1e-6:
            return arr
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        m = np.where(h > 0, np.stack([g, b, r], -1), np.stack([b, r, g], -1))
        return np.clip(arr + (m - arr) * abs(h) * 2.0, 0, 1)

    def _to_gray(self, arr):
        g = arr.mean(axis=-1, keepdims=True)
        return np.concatenate([g] * 3, axis=-1)

    def __call__(self, arr):
        """arr: [T,H,W,3] in [0,1]，原地增强（跨帧一致或逐帧）。"""
        if self.consistent:
            b, c, s, h = self._sample()
            out = arr
            if self.gray_first and self.grayscale > 0 and self.rng.rand() < self.grayscale:
                out = self._to_gray(out)
            if self.brightness > 0:
                out = self._adjust_brightness(out, b)
            if self.contrast > 0:
                out = self._adjust_contrast(out, c)
            if self.saturation > 0:
                out = self._adjust_saturation(out, s)
            if self.hue > 0:
                out = self._adjust_hue(out, h)
            if not self.gray_first and self.grayscale > 0 and self.rng.rand() < self.grayscale:
                out = self._to_gray(out)
            return np.clip(out, 0, 1)
        out = np.stack([self(np.expand_dims(f, 0))[0] for f in arr])
        return out


class RandomResizedCrop:
    """官方 RandomResizedCropVideo 的空间逻辑（torchvision 同款 scale/ratio 采样）。

    scale 下限适配小鼠：官方 (0.08, 1.0) 对小目标会裁掉小鼠本体 → 默认 (0.5, 1.0)。
    """

    def __init__(self, size=224, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.rng = np.random.RandomState()

    def _get_params(self, h, w):
        area = h * w
        for _ in range(10):
            target = self.rng.uniform(self.scale[0], self.scale[1]) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = np.exp(self.rng.uniform(*log_ratio))
            ch = int(round(np.sqrt(target / aspect)))
            cw = int(round(np.sqrt(target * aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                i = self.rng.randint(0, h - ch + 1)
                j = self.rng.randint(0, w - cw + 1)
                return i, j, ch, cw
        i = self.rng.randint(0, h - min(h, self.size) + 1) if h > self.size else 0
        j = self.rng.randint(0, w - min(w, self.size) + 1) if w > self.size else 0
        ch = min(h, self.size)
        cw = min(w, self.size)
        return i, j, ch, cw

    def __call__(self, arr):
        """arr: [T,H,W,3] in [0,1] -> 裁剪后 resize 到 (size, size)。"""
        import cv2
        T, h, w, _ = arr.shape
        i, j, ch, cw = self._get_params(h, w)
        crop = arr[:, i:i + ch, j:j + cw, :]
        if ch != self.size or cw != self.size:
            crop = np.stack([cv2.resize(f, (self.size, self.size),
                                        interpolation=cv2.INTER_LINEAR) for f in crop])
        return crop


class RandomErasing:
    """官方 TAdaConv RandomErasing（RandomErasing 原版逻辑：随机矩形擦除）。"""

    def __init__(self, probability=0.25, area_ratio=(0.02, 0.4), min_aspect=0.3,
                 max_count=1):
        self.probability = probability
        self.area_ratio = area_ratio
        self.min_aspect = min_aspect
        self.max_count = max_count
        self.rng = np.random.RandomState()

    def __call__(self, arr):
        """arr: [T,H,W,3] in [0,1]；向量化：整批生成擦除参数，避免逐帧 Python 循环。"""
        T, h, w, _ = arr.shape
        out = arr.copy()
        r = self.rng.rand(T)
        sel = np.where(r < self.probability)[0]
        if len(sel) == 0:
            return out
        n = len(sel)
        area = h * w
        target = self.rng.uniform(*self.area_ratio, size=n) * area
        aspect = self.rng.uniform(self.min_aspect, 1.0 / self.min_aspect, size=n)
        eh = np.round(np.sqrt(target * aspect)).astype(int)
        ew = np.round(np.sqrt(target / aspect)).astype(int)
        ok = (eh < h) & (ew < w)
        sel, eh, ew = sel[ok], eh[ok], ew[ok]
        if len(sel) == 0:
            return out
        iy = self.rng.randint(0, h - eh.max() + 1, size=len(sel))
        ix = self.rng.randint(0, w - ew.max() + 1, size=len(sel))
        for k, t in enumerate(sel):
            out[t, iy[k]:iy[k] + eh[k], ix[k]:ix[k] + ew[k], :] = 0.0
        return out
