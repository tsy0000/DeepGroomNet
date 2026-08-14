"""
ClipDataset：把 (video, start, end, label) 转成 (C,T,H,W) 归一化张量。

对齐规范：
  - 帧号 = sample_frames(start,end,n_frames) 均匀抽取，训练/推理一致。
  - 归一化用数据集实测 mean/std（config.CLIP.mean/std）。
  - 数据增强（仅 train）clip 一致：随机水平翻转 + 随机缩放裁剪 + 色彩抖动。
  - homecage 中间域：clips 的 video 字段即 clip 路径，start/end 由 homecage_clips 给出。
"""
from __future__ import annotations

import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.clip_sampler import sample_frames, read_frames, jitter_window
from data.pose_target import SUPERANIMAL_SKELETONS


def _resize_frame(fr, size, interp=cv2.INTER_LINEAR):
    return cv2.resize(fr, (size, size), interpolation=interp)


def add_motion_channel(arr):
    """arr: [T,H,W,3] in [0,1] -> [T,H,W,4]（第 4 通道=帧间绝对差灰度）。

    Sakamoto 2022 帧差输入 / DeepEthogram 光流流 / BehaveAI color-from-motion 的
    模块化实现；训练（dataset）与推理（infer/scripts）共用，保证口径一致。
    """
    d = np.abs(np.diff(arr, axis=0, prepend=arr[:1])).mean(axis=-1, keepdims=True)
    return np.concatenate([arr, d], axis=-1)


class ClipDataset(Dataset):
    def __init__(self, clips, cfg, split="train"):
        """
        clips: list of dict {video, start, end, label}
        cfg  : config.yaml 的 dict
        split: train -> 增强 + 边界抖动；val/test -> 确定性中心处理（不增强）
        """
        self.clips = clips
        self.cfg = cfg
        self.split = split
        self.train = split == "train"
        self.n_frames = int(cfg["CLIP"]["num_input_frames"])
        self.size = int(cfg["CLIP"]["target_size"])
        self.jit = int(cfg["CLIP"].get("jitter_frames", 4)) if self.train else 0
        self.crop_scale = tuple(cfg["CLIP"].get("aug_crop_scale", (0.5, 1.0)))
        self.cj_brightness = float(cfg["CLIP"].get("aug_brightness", 0.4))
        self.cj_contrast = float(cfg["CLIP"].get("aug_contrast", 0.4))
        self.cj_saturation = float(cfg["CLIP"].get("aug_saturation", 0.4))
        self.cj_hue = float(cfg["CLIP"].get("aug_hue", 0.0))
        self.re_prob = float(cfg["CLIP"].get("aug_erase_prob", 0.25))
        self.motion = bool(cfg["CLIP"].get("motion_channel", False))
        self.flow = bool(cfg["CLIP"].get("flow_channel", False))
        self.flow_root = cfg["CLIP"].get("flow_root", "") or ""
        self.in_ch = 3 + (1 if self.motion else 0) + (2 if self.flow else 0)
        mean = cfg["CLIP"]["mean"]
        std = cfg["CLIP"]["std"]
        if self.motion:
            mean = list(mean) + [cfg["CLIP"].get("motion_mean", 0.02)]
            std = list(std) + [cfg["CLIP"].get("motion_std", 0.05)]
        if self.flow:
            from data.flow_utils import FLOW_MEAN, FLOW_STD
            mean = list(mean) + [FLOW_MEAN, FLOW_MEAN]
            std = list(std) + [FLOW_STD, FLOW_STD]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1)
        self.kp_bank = None
        self.kp_use_parts = None
        self.kp_min_lik = 0.5
        self.kp_render = False
        self.kp_sigma = 0.6
        self.kp_hm = 64
        self._last_flip = False
        kp_cfg = cfg.get("KEYPOINTS") or {}
        if kp_cfg.get("enabled", True) and kp_cfg.get("root"):
            from data.keypoints import KeypointBank
            self.kp_bank = KeypointBank(kp_cfg["root"])
            self.kp_use_parts = kp_cfg.get("use_parts") or None
            self.kp_min_lik = float(kp_cfg.get("min_likelihood", 0.5))
            self.kp_render = bool(kp_cfg.get("render_prior", True))
            self.kp_sigma = float(kp_cfg.get("render_sigma", 0.6))
            self.kp_hm = int(kp_cfg.get("render_heatmap_size", 64))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        if "clip" in c and c["clip"]:
            total_c = self._clip_total(c["clip"])
            indices = sample_frames(0, max(total_c, 1), self.n_frames)
            if len(indices) < self.n_frames and indices:
                indices = (np.resize(np.asarray(indices), self.n_frames)).tolist()
            frames = self._read_clip_frames(c["clip"], indices)
            frames = [f if f is not None else np.zeros((480, 480, 3), np.uint8)
                      for f in frames]
            while len(frames) < self.n_frames:
                frames.append(frames[len(frames) % max(1, len(frames))])
            arr = np.stack([_resize_frame(f, self.size) for f in frames]).astype(np.float32) / 255.0
            if self.train:
                arr = self._augment(arr)
            abs_frames = [int(c.get("start", 0)) + i for i in indices]
            pose, kp_map = self._pose(c, abs_frames)
            kp_map = self._finalize_kp(arr, kp_map)
            flow = self._flow(c)
            arr, pose, kp_map = self._sanitize(arr, pose, kp_map, c)
            return (self._finalize(arr, flow), pose, kp_map,
                    torch.tensor(self.class_to_idx(c["label"]), dtype=torch.long))

        start, end = c["start"], c["end"]
        if self.jit > 0:
            start, end = jitter_window(start, end, self._video_total(c["video"]),
                                       self.jit, np.random.default_rng())
        indices = sample_frames(start, end, self.n_frames)
        if len(indices) < self.n_frames and len(indices) > 0:
            indices = (np.resize(np.asarray(indices), self.n_frames)).tolist()
        frames = read_frames(c["video"], indices)
        if not frames:
            frames = [np.zeros((480, 480, 3), np.uint8)] * self.n_frames

        arr = np.stack([_resize_frame(f, self.size) for f in frames]).astype(np.float32) / 255.0
        if self.train:
            arr = self._augment(arr)
        pose, kp_map = self._pose(c, indices)
        kp_map = self._finalize_kp(arr, kp_map)
        flow = self._flow(c)
        arr, pose, kp_map = self._sanitize(arr, pose, kp_map, c)
        return (self._finalize(arr, flow), pose, kp_map,
                torch.tensor(self.class_to_idx(c["label"]), dtype=torch.long))

    def _flow(self, c, n_out=None):
        """加载 clip 光流（[n_sampled,2,32,32] -> [n_out,2,size,size] 归一化）；无则 None。

        光流 npz 由 scripts/13_gen_flow.py 按 CLIP.num_input_frames 生成（与帧采样
        严格对齐，TinyMotionNet 官方权重）。帧数必须匹配模型输入——不匹配直接报错，
        杜绝静默插值/上采样造成的"将就"（训练与推理分布漂移）。
        """
        if not self.flow or not self.flow_root:
            return None
        from data.flow_utils import load_clip_flow, flow_to_input
        clip_path = c.get("clip") or c.get("video", "")
        fl = load_clip_flow(clip_path, self.flow_root)
        if fl is None:
            return None
        n_out = n_out or self.n_frames
        if fl.shape[0] != n_out:
            raise RuntimeError(
                f"光流帧数 {fl.shape[0]} != 模型输入 {n_out}（{clip_path}）。"
                f"请按 CLIP.num_input_frames 重新生成光流：python scripts/13_gen_flow.py")
        return flow_to_input(fl, self.size).numpy()

    def _kp_mouse(self, c):
        """关键点按小鼠命名：预裁剪 clip 取文件名前缀；原始视频取父目录名。"""
        if "clip" in c and c["clip"]:
            return os.path.basename(c["clip"]).split("_")[0]
        return os.path.basename(os.path.dirname(c["video"]))

    def _pose(self, c, abs_frames):
        """abs_frames: 视频绝对帧号 -> (pose [T,K,3], kp_map [T,C,H,W]) 恒为张量。

        kp_map = PoseTargetGenerator 生成的 PoseC3D 伪热图（关键点+肢体，mmaction2 原版）；
        无关键点时：pose=全零（valid=0）、kp_map=全一（中性先验）——
        保证 DataLoader default_collate 永不遇到 None。
        """
        K = len(self.kp_use_parts) if self.kp_use_parts else 27
        n_c = K + len(SUPERANIMAL_SKELETONS) if self.kp_use_parts is None else K
        neutral_map = torch.ones(self.n_frames, n_c, self.kp_hm, self.kp_hm,
                                 dtype=torch.float16)
        if self.kp_bank is None:
            return torch.zeros(self.n_frames, K, 3, dtype=torch.float16), neutral_map
        from data.keypoints import pose_features
        from data.pose_target import PoseTargetGenerator
        bps, kp = self.kp_bank.get(self._kp_mouse(c))
        if kp is None:
            return torch.zeros(self.n_frames, K, 3, dtype=torch.float16), neutral_map.half()
        p = pose_features(kp, abs_frames, bps, use_parts=self.kp_use_parts,
                          min_lik=self.kp_min_lik)
        pose = (torch.from_numpy(p) if p is not None else torch.zeros(
            self.n_frames, K, 3, dtype=torch.float32)).half()
        if not self.kp_render:
            return pose, neutral_map
        gen = PoseTargetGenerator(sigma=float(self.kp_sigma))
        skeletons = SUPERANIMAL_SKELETONS if self.kp_use_parts is None else None
        maps = gen.gen(pose.numpy(), self.kp_hm, skeletons=skeletons)
        return pose.half(), torch.from_numpy(maps).half()

    def _finalize_kp(self, arr, kp_map):
        """对 kp_map 施加与图像一致的增强（翻转 + 旋转，几何严格同步）。

        旋转增强（Sakamoto 2022）后必须同步旋转关键点热图，否则 pose 分支学到
        "图像转/热图不转"的错误映射（几何一致性是 PoseC3D 式分支正确性的前提）。
        """
        if kp_map is None:
            return kp_map
        if getattr(self, "_last_flip", False):
            kp_map = torch.flip(kp_map, dims=[-1])
        rot = getattr(self, "_last_rotate", 0.0)
        if abs(rot) > 1e-6 and kp_map.ndim == 5:
            from scipy.ndimage import rotate as _rot
            kp_map = torch.from_numpy(
                _rot(kp_map.numpy(), rot, axes=(3, 4), reshape=False, order=1,
                     mode="nearest")).to(kp_map.dtype)
        return kp_map

    def _finalize(self, arr, flow=None):
        """arr: [T,H,W,3] in [0,1] -> 归一化张量 [C,T,H,W]（帧差/光流可选通道）。"""
        chans = [arr]
        if self.motion:
            d = np.abs(np.diff(arr, axis=0, prepend=arr[:1])).mean(axis=-1, keepdims=True)
            chans.append(d)
        if self.flow:
            if flow is None:
                flow = np.zeros((self.n_frames, 2, self.size, self.size), np.float32)
            chans.append(np.transpose(flow, (0, 2, 3, 1)))
        x = torch.from_numpy(np.transpose(np.concatenate(chans, axis=-1), (3, 0, 1, 2)))
        return (x - self.mean) / self.std

    def _video_total(self, path):
        total = getattr(self, "_total_cache", None)
        if total is None or total[0] != path:
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._total_cache = (path, n)
        return self._total_cache[1]

    @staticmethod
    def _clip_total(path):
        import cv2
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    @staticmethod
    def _read_clip_frames(path, indices):
        """只 seek 读指定帧号（比全读 45 帧快 ~4 倍）。"""
        import cv2
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = []
        for i in indices:
            if i >= total or i < 0:
                out.append(None)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            out.append(fr if ok else None)
        cap.release()
        return out

    @staticmethod
    def _read_clipfile(path):
        """seek 读取整段小 clip（45 帧 mp4），避免 OpenCV 整段解码的内存峰值。"""
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        return frames

    def class_to_idx(self, label: str) -> int:
        names = self.cfg["CLASSES"]["openfield_keep"]
        return names.index(label) if label in names else names.index("background")

    def _augment(self, arr):
        self._last_flip = False
        if np.random.rand() < 0.5:
            arr = arr[:, :, ::-1, :]
            self._last_flip = True
        from data.augmentations import RandomResizedCrop, ColorJitter, RandomErasing
        if self.train:
            arr = RandomResizedCrop(size=self.size,
                                    scale=self.crop_scale)(arr)
        if self.train:
            arr = ColorJitter(brightness=self.cj_brightness,
                              contrast=self.cj_contrast,
                              saturation=self.cj_saturation,
                              hue=self.cj_hue,
                              consistent=True)(arr)
        if self.train:
            arr = RandomErasing(probability=self.re_prob)(arr)
        self._last_rotate = 0.0
        if self.train and float(self.cfg["CLIP"].get("aug_rotate_deg", 0.0)) > 0:
            deg = float(self.cfg["CLIP"]["aug_rotate_deg"]) * (2 * np.random.rand() - 1)
            self._last_rotate = deg
            from scipy.ndimage import rotate
            arr = rotate(arr, deg, axes=(1, 2), reshape=False, order=1,
                         mode="nearest").astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    def _sanitize(self, arr, pose, kp_map, c):
        """毒样本守卫：NaN/Inf 帧/关键点 → 填 0 并打印定位（防训练 loss 变 nan）。"""
        if arr is not None and (not np.isfinite(arr).all()):
            print(f"  [poison] x NaN/Inf: {c.get('video', c.get('clip', '?'))}@{c.get('start')}",
                  flush=True)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if pose is not None and isinstance(pose, torch.Tensor) and not torch.isfinite(pose).all():
            print(f"  [poison] pose NaN/Inf: {c.get('video', c.get('clip', '?'))}@{c.get('start')}",
                  flush=True)
            pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        if kp_map is not None and isinstance(kp_map, torch.Tensor) and not torch.isfinite(kp_map).all():
            print(f"  [poison] kp_map NaN/Inf: {c.get('video', c.get('clip', '?'))}@{c.get('start')}",
                  flush=True)
            kp_map = torch.nan_to_num(kp_map, nan=0.0, posinf=0.0, neginf=0.0)
        return arr, pose, kp_map
