"""小鼠身体掩码（零标注，经典背景减除法）。

方法来源：
  - Jhuang et al., "Vision-Based Recognition of Mice Home-Cage Behaviors" (VAIB'10)：
    静态相机下用**中值帧**估计背景，逐帧做背景减除得到前景掩码。
  - MIT CBMM 小鼠识别系统：foreground mask 后接形态学清理 + 最大连通域。
  - DeepEthogram (Bohnslav 2021)：不做掩码，但明确把"背景像素噪声"列为难点；
    我们在注意力模块用掩码先验（门只作用在小鼠像素上）规避背景注意力浪费。

对预裁剪 clip（YOLO 裁剪、单只小鼠、静止相机）：
  1) 灰度时均值 -> 中值背景 bg（45 帧内小鼠基本不动，中值帧=笼具+静止小鼠体）
  2) fg = |frame - bg| 的时均值（运动的前爪/嘴部/尾巴等区域高响应）
  3) 自适应阈值（均值+k·std）二值化 -> 形态学闭运算填洞 -> 取最大连通域
  4) 膨胀恢复身体边缘，可选与"中心先验"并集（YOLO 裁剪保证小鼠在画面中央）
输出：0/1 掩码 [H, W] float32，1 = 小鼠（前景），0 = 背景。
"""
import numpy as np
import cv2


def median_background(gray):
    """gray: [T,H,W] float32 -> 中值背景帧 [H,W]。"""
    return np.median(gray, axis=0)


def compute_mouse_mask(frames, k_thresh=0.35, dilate_k=7, use_center_union=True,
                       center_margin=0.12, min_coverage=0.05, max_coverage=0.9):
    """frames: [T,H,W,3] uint8 或 float32 -> mask [H,W] float32（1=小鼠）。

    参数：
      k_thresh       前景阈值：fg > mean(fg) + k_thresh * std(fg)
      dilate_k       形态学膨胀核大小（恢复身体边缘）
      use_center_union 与中心矩形取并集（YOLO 裁剪保证小鼠居中，兜底漏检）
      min/max_coverage 掩码面积校验（异常时回退中心矩形）
    """
    gray = frames.mean(axis=3).astype(np.float32) / 255.0 if frames.max() > 1.0 else \
        frames.mean(axis=3).astype(np.float32)
    T = gray.shape[0]
    if T < 3:
        h, w = gray.shape[1:]
        return _center_mask(h, w, center_margin)

    bg = median_background(gray)
    diff = np.abs(gray - bg[None]).mean(axis=0)
    th = diff.mean() + k_thresh * (diff.std() + 1e-6)
    mask = (diff > th).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(np.argmax(areas)) + 1
        mask = (lab == largest).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((dilate_k, dilate_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cov = mask.mean()
    h, w = mask.shape
    center = _center_mask(h, w, center_margin)
    if use_center_union:
        mask = np.maximum(mask, center).astype(np.uint8)
    if cov < min_coverage or cov > max_coverage:
        mask = center
    return mask.astype(np.float32)


def _center_mask(h, w, margin=0.12):
    m = np.zeros((h, w), np.float32)
    y0, y1 = int(h * margin), int(h * (1 - margin))
    x0, x1 = int(w * margin), int(w * (1 - margin))
    m[y0:y1, x0:x1] = 1.0
    return m


def mask_coverage_ratio(mask):
    return float(mask.mean())
