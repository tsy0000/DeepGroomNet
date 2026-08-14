"""
推理端 ROI 裁剪（修复训练/推理不一致的关键组件）。

问题：训练 clip 由 scripts/02_gen_clips.py 用 YOLO 检测窗口首/中/末 3 帧 → 并集框 + 1.2 外扩
→ 裁剪 resize 224 生成；而旧版推理直接整帧 resize 224 —— 对 datasetssap3 原始画幅（586~1008px）
视频，小鼠只占画面一小部分，推理输入与训练分布不一致。

方案：推理端用与 02 完全相同的策略，对每个推理窗口动态检测裁剪框（窗口首/中/末 3 帧并集框），
保证训练/推理 ROI 分布一致。YOLO 权重路径取 config.INFER.yolo_weights（缺省回退到
C:\\Users\\WIN10\\PycharmProjects\\mouseaction\\DeepBehavior\\yolo11n.pt，与原项目一致）。
"""
from __future__ import annotations

import os

import cv2
import numpy as np


class WindowCropBoxer:
    """与 scripts/02_gen_clips.py 同策略的窗口级裁剪器。

    对每个窗口：检测首/中/末 3 帧（同一 YOLO、同一 conf/expand），取并集框；
    无检测时回退窗口中心最大方形（02 回退为全帧中心方形）。
    """

    def __init__(self, weights=None, conf=0.1, expand=1.2, device=None):
        self.conf = conf
        self.expand = expand
        self._weights = weights
        self._yolo = None
        if device is None:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        self._device = device

    @property
    def yolo(self):
        if self._yolo is None:
            if not self._weights or not os.path.exists(self._weights):
                self._weights = r"C:\Users\WIN10\PycharmProjects\mouseaction\DeepBehavior\yolo11n.pt"
            if not os.path.exists(self._weights):
                raise FileNotFoundError(
                    f"YOLO 权重不存在: {self._weights}（请配置 config.INFER.yolo_weights）")
            from ultralytics import YOLO
            self._yolo = YOLO(self._weights)
        return self._yolo

    def _union_box(self, boxes, size):
        if not boxes:
            return None
        xs1 = min(b[0] for b in boxes)
        ys1 = min(b[1] for b in boxes)
        xs2 = max(b[2] for b in boxes)
        ys2 = max(b[3] for b in boxes)
        w, h = xs2 - xs1, ys2 - ys1
        cx, cy = (xs1 + xs2) / 2, (ys1 + ys2) / 2
        nw, nh = w * self.expand, h * self.expand
        x1 = max(0, int(cx - nw / 2))
        y1 = max(0, int(cy - nh / 2))
        x2 = min(size[0], int(cx + nw / 2))
        y2 = min(size[1], int(cy + nh / 2))
        return (x1, y1, x2, y2)

    @staticmethod
    def _center_box(size):
        w, h = size
        s = min(w, h)
        x1 = (w - s) // 2
        y1 = (h - s) // 2
        return (x1, y1, x1 + s, y1 + s)

    def box_for_frames(self, frames):
        """检测窗口首/中/末 3 帧 → 并集框 + expand；无检测回退中心方形。"""
        h, w = frames[0].shape[:2]
        det_idx = [0, len(frames) // 2, len(frames) - 1]
        boxes = []
        for i in det_idx:
            r = self.yolo.predict(frames[i], conf=self.conf, verbose=False,
                                  device=self._device)[0]
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy[0].cpu().numpy().astype(int)
                boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return self._union_box(boxes, (w, h)) or self._center_box((w, h))

    def apply(self, frames, target_size):
        """裁剪（窗口级检测框）并 resize 到 target_size。与 02 _crop_resize 一致。"""
        box = self.box_for_frames(frames)
        out = []
        x1, y1, x2, y2 = box
        for fr in frames:
            c = fr[y1:y2, x1:x2]
            out.append(cv2.resize(c, (target_size, target_size),
                                  interpolation=cv2.INTER_AREA))
        return out


def build_crop_boxer(cfg):
    """按 config.INFER.crop_mode 构建裁剪器；none 返回 None。"""
    mode = (cfg.get("INFER", {}).get("crop_mode") or "window").lower()
    if mode in ("none", "off"):
        return None
    if mode not in ("window", "static"):
        raise ValueError(f"未知 crop_mode: {mode}（可选 none | window）")
    weights = cfg.get("INFER", {}).get("yolo_weights", "") or None
    conf = float(cfg["CLIP"].get("yolo_conf", 0.1))
    expand = float(cfg["CLIP"].get("yolo_expand", 1.2))
    if mode == "static":
        return StaticCropBoxer(weights=weights, conf=conf, expand=expand)
    return WindowCropBoxer(weights=weights, conf=conf, expand=expand)


class StaticCropBoxer(WindowCropBoxer):
    """静态框模式：全视频首/中/末帧检测一次，整段共用（更快，框可能跟不上小鼠移动）。"""

    def __init__(self, weights=None, conf=0.1, expand=1.2):
        super().__init__(weights=weights, conf=conf, expand=expand)
        self._static_box = None

    def apply(self, frames, target_size):
        if self._static_box is None:
            h, w = frames[0].shape[:2]
            det_idx = [0, len(frames) // 2, len(frames) - 1]
            boxes = []
            for i in det_idx:
                r = self.yolo.predict(frames[i], conf=self.conf, verbose=False,
                                      device=self._device)[0]
                if r.boxes is not None and len(r.boxes):
                    b = r.boxes.xyxy[0].cpu().numpy().astype(int)
                    boxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
            self._static_box = self._union_box(boxes, (w, h)) or self._center_box((w, h))
        x1, y1, x2, y2 = self._static_box
        out = []
        for fr in frames:
            c = fr[y1:y2, x1:x2]
            out.append(cv2.resize(c, (target_size, target_size),
                                  interpolation=cv2.INTER_AREA))
        return out
