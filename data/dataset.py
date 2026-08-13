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
    d = np.abs(np.diff(arr, axis=0, prepend=arr[:1])).mean(axis=-1, keepdims=True)
    return np.concatenate([arr, d], axis=-1)

class ClipDataset(Dataset):

    def __init__(self, clips, cfg, split='train'):
        self.clips = clips
        self.cfg = cfg
        self.split = split
        self.train = split == 'train'
        self.n_frames = int(cfg['CLIP']['num_input_frames'])
        self.size = int(cfg['CLIP']['target_size'])
        self.jit = int(cfg['CLIP'].get('jitter_frames', 4)) if self.train else 0
        self.motion = bool(cfg['CLIP'].get('motion_channel', False))
        self.flow = bool(cfg['CLIP'].get('flow_channel', False))
        self.flow_root = cfg['CLIP'].get('flow_root', '') or ''
        self.in_ch = 3 + (1 if self.motion else 0) + (2 if self.flow else 0)
        mean = cfg['CLIP']['mean']
        std = cfg['CLIP']['std']
        if self.motion:
            mean = list(mean) + [cfg['CLIP'].get('motion_mean', 0.02)]
            std = list(std) + [cfg['CLIP'].get('motion_std', 0.05)]
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
        kp_cfg = cfg.get('KEYPOINTS') or {}
        if kp_cfg.get('enabled', True) and kp_cfg.get('root'):
            from data.keypoints import KeypointBank
            self.kp_bank = KeypointBank(kp_cfg['root'])
            self.kp_use_parts = kp_cfg.get('use_parts') or None
            self.kp_min_lik = float(kp_cfg.get('min_likelihood', 0.5))
            self.kp_render = bool(kp_cfg.get('render_prior', True))
            self.kp_sigma = float(kp_cfg.get('render_sigma', 0.6))
            self.kp_hm = int(kp_cfg.get('render_heatmap_size', 64))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        if 'clip' in c and c['clip']:
            frames = self._read_clipfile(c['clip'])
            indices = sample_frames(0, len(frames), self.n_frames)
            if len(indices) < self.n_frames and indices:
                indices = np.resize(np.asarray(indices), self.n_frames).tolist()
            sel = []
            for i in indices:
                if 0 <= i < len(frames):
                    sel.append(frames[i])
                else:
                    sel.append(np.zeros((480, 480, 3), np.uint8))
            frames = sel or [np.zeros((480, 480, 3), np.uint8)] * self.n_frames
            while len(frames) < self.n_frames:
                frames.append(frames[len(frames) % max(1, len(frames))])
            arr = np.stack([_resize_frame(f, self.size) for f in frames]).astype(np.float32) / 255.0
            if self.train:
                arr = self._augment(arr)
            abs_frames = [int(c.get('start', 0)) + i for i in indices]
            (pose, kp_map) = self._pose(c, abs_frames)
            kp_map = self._finalize_kp(arr, kp_map)
            flow = self._flow(c)
            return (self._finalize(arr, flow), pose, kp_map, torch.tensor(self.class_to_idx(c['label']), dtype=torch.long))
        (start, end) = (c['start'], c['end'])
        if self.jit > 0:
            (start, end) = jitter_window(start, end, self._video_total(c['video']), self.jit, np.random.default_rng())
        indices = sample_frames(start, end, self.n_frames)
        if len(indices) < self.n_frames and len(indices) > 0:
            indices = np.resize(np.asarray(indices), self.n_frames).tolist()
        frames = read_frames(c['video'], indices)
        if not frames:
            frames = [np.zeros((480, 480, 3), np.uint8)] * self.n_frames
        arr = np.stack([_resize_frame(f, self.size) for f in frames]).astype(np.float32) / 255.0
        if self.train:
            arr = self._augment(arr)
        (pose, kp_map) = self._pose(c, indices)
        kp_map = self._finalize_kp(arr, kp_map)
        flow = self._flow(c)
        return (self._finalize(arr, flow), pose, kp_map, torch.tensor(self.class_to_idx(c['label']), dtype=torch.long))

    def _flow(self, c):
        if not self.flow or not self.flow_root:
            return None
        from data.flow_utils import load_clip_flow, flow_to_input
        clip_path = c.get('clip') or c.get('video', '')
        fl = load_clip_flow(clip_path, self.flow_root)
        if fl is None:
            return None
        return flow_to_input(fl, self.size).numpy()

    def _kp_mouse(self, c):
        if 'clip' in c and c['clip']:
            return os.path.basename(c['clip']).split('_')[0]
        return os.path.basename(os.path.dirname(c['video']))

    def _pose(self, c, abs_frames):
        K = len(self.kp_use_parts) if self.kp_use_parts else 27
        n_c = K + len(SUPERANIMAL_SKELETONS) if self.kp_use_parts is None else K
        neutral_map = torch.ones(self.n_frames, n_c, self.kp_hm, self.kp_hm, dtype=torch.float32)
        if self.kp_bank is None:
            return (torch.zeros(self.n_frames, K, 3, dtype=torch.float32), neutral_map)
        from data.keypoints import pose_features
        from data.pose_target import PoseTargetGenerator
        (bps, kp) = self.kp_bank.get(self._kp_mouse(c))
        if kp is None:
            return (torch.zeros(self.n_frames, K, 3, dtype=torch.float32), neutral_map)
        p = pose_features(kp, abs_frames, bps, use_parts=self.kp_use_parts, min_lik=self.kp_min_lik)
        pose = torch.from_numpy(p) if p is not None else torch.zeros(self.n_frames, K, 3, dtype=torch.float32)
        if not self.kp_render:
            return (pose, neutral_map)
        gen = PoseTargetGenerator(sigma=float(self.kp_sigma))
        skeletons = SUPERANIMAL_SKELETONS if self.kp_use_parts is None else None
        maps = gen.gen(pose.numpy(), self.kp_hm, skeletons=skeletons)
        return (pose, torch.from_numpy(maps))

    def _finalize_kp(self, arr, kp_map):
        if kp_map is not None and getattr(self, '_last_flip', False):
            kp_map = torch.flip(kp_map, dims=[-1])
        return kp_map

    def _finalize(self, arr, flow=None):
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
        total = getattr(self, '_total_cache', None)
        if total is None or total[0] != path:
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self._total_cache = (path, n)
        return self._total_cache[1]

    @staticmethod
    def _read_clipfile(path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            (ok, fr) = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        return frames

    def class_to_idx(self, label: str) -> int:
        names = self.cfg['CLASSES']['openfield_keep']
        return names.index(label) if label in names else names.index('background')

    def _augment(self, arr):
        self._last_flip = False
        if np.random.rand() < 0.5:
            arr = arr[:, :, ::-1, :]
            self._last_flip = True
        scale = np.random.uniform(0.9, 1.1)
        hs = ws = int(round(self.size * scale))
        if hs != self.size:
            arr = np.stack([cv2.resize(f, (ws, hs), interpolation=cv2.INTER_LINEAR) for f in arr]).astype(np.float32)
        if hs > self.size:
            i = np.random.randint(0, hs - self.size + 1)
            j = np.random.randint(0, ws - self.size + 1)
            arr = arr[:, i:i + self.size, j:j + self.size, :]
        elif hs < self.size:
            pad = (self.size - hs) // 2
            arr = np.pad(arr, ((0, 0), (pad, self.size - hs - pad), (pad, self.size - hs - pad), (0, 0)), mode='edge')
        jit = np.random.uniform(0.85, 1.15, size=3).astype(np.float32)
        arr = arr * jit
        return np.clip(arr, 0.0, 1.0)
