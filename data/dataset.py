"""DeepGroomNet_v3 - 数据管线（对齐 DeepEthogram 帧级范式）。

- 帧级标签: 每个视频有逐帧 one-hot 标签
- 实时解码: 每个 clip 从视频读固定窗口帧 (vidio.VideoReader 式), 随机起点
- clip 标签 = 窗口中心/某帧标签
- 训练/验证按"视频/小鼠"划分 (跨个体泛化)
- 增强: DeepEthogram 官方 (hflip 0.5 + 旋转 + ColorJitter + RandomGrayscale + Normalize)

关键设计:
  - 每个 sample = 一个 (video, 随机起点) → 读 frames_per_clip 帧
  - 类别不平衡靠 pos_weight BCE 处理 (不用采样器)
"""
import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class FrameLevelDataset(Dataset):
    """帧级行为分类数据集 (DeepEthogram 范式)。

    videos: list of {video, labels, n_frames, mouse}
      labels: [n_frames] int (类索引)
    cfg: config dict

    帧缓存优化: 每个视频首次访问时全解码存入内存 (连续读极快 0.1ms/帧,
    vs 随机 seek 8.8ms/帧)。内存不足时退回按需 seek。
    """
    def __init__(self, videos, cfg, split="train", class_names=None, frame_window=16,
                 random_start=True, clips_per_video=None, fixed_windows=None):
        self.videos = videos
        self.cfg = cfg
        self.split = split
        self.train = split == "train"
        self.class_names = class_names or cfg["CLASSES"]["openfield"]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.frame_window = frame_window
        self.random_start = random_start and self.train
        # 类平衡采样: 预计算的 (video_idx, start) 窗口列表 (中心帧=指定类),
        # 提供时训练模式直接用固定窗口, 保证逐类均匀采样 (修复过采样失效).
        self.fixed_windows = fixed_windows
        self.n_frames = frame_window
        self.size = int(cfg["CLIP"]["target_size"])
        self.in_ch = 3  # 固定 3 通道 RGB (3D 骨干已在内部建模运动, 无需帧差通道)
        # DeepEthogram/官方对齐: 预训练 mean/std (TAdaConv finetune.yaml MEAN=0.45, STD=0.225)
        # 输入已 /255 到 [0,1], 这里做 (x-mean)/std
        self.mean = float(cfg["CLIP"].get("norm_mean", 0.45))
        self.std = float(cfg["CLIP"].get("norm_std", 0.225))
        # 每视频每 epoch 抽取的 clip 数 (限制 epoch 大小, 帧级分类标准做法)
        self.clips_per_video = clips_per_video or int(cfg["TRAIN"].get("clips_per_video", 200))
        self.stride = int(cfg["TRAIN"].get("val_stride", 8))
        self._caps = {}  # video -> VideoCapture
        # YOLO 裁剪 (成熟预处理): 从 yolo_boxes.json 读每视频稳定裁剪框, 裁剪放大到 target_size
        self.yolo_boxes = None
        if cfg["CLIP"].get("yolo_crop", False):
            import json as _json
            bp = os.path.join(cfg["DATA"]["meta_root"], "yolo_boxes.json")
            if os.path.exists(bp):
                self.yolo_boxes = _json.load(open(bp, encoding="utf-8"))
        # val/test: 按类别分层预构建索引, 每类取 samples_per_class, 保证稀有类覆盖
        if not self.train:
            self._build_class_index()

    def _build_class_index(self):
        """val/test 分层索引: 每类均匀采样, 保证 FG/BG 被评估。"""
        import random as _r
        per_class = int(self.cfg["TRAIN"].get("val_samples_per_class", 200))
        self._class_idx = {}
        self._class_pool = {}
        for i in range(len(self.class_names)):
            pool = []
            for vi, v in enumerate(self.videos):
                labels = np.asarray(v["labels"])
                centers = np.where(labels == i)[0]
                # 每视频为该类贡献窗口起点
                for c in centers:
                    s = max(0, min(int(c) - self.frame_window // 2,
                                   v["n_frames"] - self.frame_window))
                    pool.append((vi, s))
            _r.seed(42)
            sample = _r.sample(pool, min(per_class, len(pool))) if pool else []
            self._class_pool[i] = sample
        self._class_count = {i: len(self._class_pool[i]) for i in range(len(self.class_names))}

    def __len__(self):
        if self.train:
            if self.fixed_windows is not None:
                return len(self.fixed_windows)
            return len(self.videos) * self.clips_per_video
        return sum(self._class_count.values())

    def _get_cap(self, video_path):
        if video_path not in self._caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"cannot open {video_path}")
            self._caps[video_path] = cap
        return self._caps[video_path]

    def _read_frames(self, video_path, start, n):
        cap = self._get_cap(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for i in range(n):
            ret, fr = cap.read()
            if not ret:
                fr = frames[-1].copy() if frames else np.zeros((self.size, self.size, 3), np.uint8)
            frames.append(fr)
        return frames

    def _sample(self, v):
        """train 随机起点。"""
        n = v["n_frames"]
        if n <= self.frame_window:
            return 0
        return random.randint(0, n - self.frame_window)

    def __getitem__(self, idx):
        if self.train:
            if self.fixed_windows is not None:
                vi, start = self.fixed_windows[idx]
                v = self.videos[vi]
                labels = np.asarray(v["labels"])
            else:
                v = self.videos[idx // self.clips_per_video]
                labels = np.asarray(v["labels"])
                start = self._sample(v)
        else:
            # val: 分层索引 -> (video_idx, start)
            for i in range(len(self.class_names)):
                n = self._class_count[i]
                if idx < n:
                    vi, start = self._class_pool[i][idx]
                    v = self.videos[vi]
                    labels = np.asarray(v["labels"])
                    break
                idx -= n
            else:
                i = len(self.class_names) - 1
                vi, start = self._class_pool[i][0]
                v = self.videos[vi]
                labels = np.asarray(v["labels"])
        # 读帧
        frames = self._read_frames(v["video"], start, self.frame_window)
        if self.yolo_boxes:
            # Swin-Panda 式: 用窗口中心帧的小鼠裁剪框, 对整个窗口一致裁剪 (时间稳定)
            center_abs = min(start + self.frame_window // 2, v["n_frames"] - 1)
            box = self._clip_box(v["video"], center_abs)
            if box is not None:
                frames = [self._crop(f, box) for f in frames]
        arr = np.stack([cv2.resize(f, (self.size, self.size)) for f in frames]).astype(np.float32) / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))  # [T,C,H,W] = [T,3,H,W]
        # 增强
        if self.train:
            arr = self._augment(arr)
        # 归一化 (官方对齐): 预训练 mean/std
        arr = (arr - self.mean) / self.std
        x = np.transpose(arr, (1, 0, 2, 3)).astype(np.float32)  # [C,T,H,W]
        # 标签 = 窗口中心帧
        center = min(start + self.frame_window // 2, v["n_frames"] - 1)
        y = int(labels[center])
        return torch.from_numpy(x), y

    def _clip_box(self, video_path, abs_frame):
        """取绝对帧 abs_frame 处的小鼠裁剪框 (兼容新旧格式)。

        新格式 (Swin-Panda): {key: {"stride": s, "boxes": [[x1,y1,x2,y2]...]}}
          boxes[k] 对应帧 k*s, 取最近采样框。
        旧格式: {key: [x1,y1,x2,y2]}  (单稳定框, 直接返回)。
        """
        key = os.path.splitext(os.path.basename(video_path))[0]
        entry = (self.yolo_boxes or {}).get(key)
        if entry is None:
            return None
        if isinstance(entry, dict) and "boxes" in entry:
            stride = int(entry["stride"])
            boxes = entry["boxes"]
            if not boxes:
                return None
            k = int(round(abs_frame / stride))
            k = min(max(k, 0), len(boxes) - 1)
            return boxes[k]
        # 旧格式: 单稳定框 [x1,y1,x2,y2]
        return entry if isinstance(entry, list) and len(entry) == 4 else None

    def _crop(self, fr, box):
        """按给定框裁剪 (clip 到边界), 返回裁剪后的帧 (未 resize)。"""
        if box is None:
            return fr
        H, W = fr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return fr
        return fr[y1:y2, x1:x2]

    def _augment(self, arr):
        """DeepEthogram 官方 GPU 增强 (arr: [T,C,H,W] float32)。"""
        c = self.cfg["CLIP"]
        # ColorJitter (brightness/contrast/saturation/hue)
        if random.random() < c.get("aug_color_p", 0.5):
            rgb = arr[:, :3]
            b = 1 + np.random.uniform(-c.get("aug_brightness", 0.25), c.get("aug_brightness", 0.25))
            cn = 1 + np.random.uniform(-c.get("aug_contrast", 0.1), c.get("aug_contrast", 0.1))
            s = 1 + np.random.uniform(-c.get("aug_saturation", 0.1), c.get("aug_saturation", 0.1))
            h = np.random.uniform(-c.get("aug_hue", 0.1), c.get("aug_hue", 0.1))
            gray = rgb.mean(axis=(2, 3), keepdims=True)
            rgb = (rgb - gray) * s + gray       # saturation
            rgb = (rgb - 0.5) * cn + 0.5        # contrast
            rgb = rgb * b                        # brightness
            arr[:, :3] = np.clip(rgb, 0, 1)
        # RandomGrayscale (0.5)
        if random.random() < c.get("aug_grayscale_p", 0.5):
            rgb = arr[:, :3].mean(axis=1, keepdims=True).repeat(3, axis=1)
            arr[:, :3] = rgb
        # Horizontal flip (0.5)
        if random.random() < c.get("aug_hflip_prob", 0.5):
            arr = arr[:, :, :, ::-1]
        # Rotation ±10°
        deg = np.random.uniform(-c.get("aug_rotation_deg", 10), c.get("aug_rotation_deg", 10))
        if abs(deg) > 0.5:
            arr = np.stack([self._rotate(im, deg) for im in arr])
        return arr

    def _rotate(self, im, deg):
        h, w = im.shape[1:3]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(im.transpose(1, 2, 0), M, (w, h),
                              flags=cv2.INTER_NEAREST).transpose(2, 0, 1)


def make_pos_weight_from_videos(videos, class_names, power=0.25):
    """统计所有视频的帧级类别分布, 算 pos_weight。"""
    from collections import Counter
    cnt = Counter()
    for v in videos:
        labels = np.asarray(v["labels"])
        for i, c in enumerate(class_names):
            cnt[c] += int((labels == i).sum())
    cc = np.array([cnt[c] for c in class_names], dtype=np.float64).clip(min=1)
    pos = cc
    neg = cc.sum() - cc
    pw = (neg / np.clip(pos, 1, None)).clip(max=100) ** power
    return torch.from_numpy(pw).float(), cc
