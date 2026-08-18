# -*- coding: utf-8 -*-
"""数据加载器 + 类别不平衡处理 (Swin-Panda 均匀过采样).

依据 Swin-Panda (Diversity 2025,17,139) 3.2 节:
  "samples from categories with fewer instances than the dataset's average are replicated.
   Uniform sampling ensures that the frame sequences of the replicated samples are not
   identical..."   (Table 1: 稀有类复制到平均值, 如 Scratching 88->157)
即: 每类窗口数低于平均值的类别, 复制(不同起始窗口, 分段均匀采样)补到平均值;
高于平均值的类别保持原量。另配 pos_weight 加权损失 + 稀有类偏置初始化。
"""
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FrameLevelDataset, make_pos_weight_from_videos


def build_oversampled_sampler(dataset, samples_per_class=None, seed=42):
    """Swin-Panda 均匀过采样采样器 (稀有类补目标, 限制 epoch 大小).

    依据 Swin-Panda (Diversity 2025,17,139) 3.2 节: 稀有类样本复制补到目标量,
    复制用不同起始窗口 -> 帧序列不重复。

    samples_per_class 为"每类目标上限" (数值测算):
      常见类(窗口数 > 目标) -> 随机下采样到目标 (保留子集, 控制 epoch 大小)
      稀有类(窗口数 < 目标) -> 复制补到目标
      每 epoch 步数 = K × samples_per_class / batch_size  (精确可控)
    若 samples_per_class 为 None, 则按论文用"有样本类的平均"作目标 (不设上限)。
    """
    import numpy as np
    import torch
    K = len(dataset.class_names)
    videos = dataset.videos
    fw = dataset.frame_window
    rng = np.random.default_rng(seed)

    pools = [[] for _ in range(K)]
    for vi, v in enumerate(videos):
        lab = np.asarray(v["labels"])
        nfr = int(v["n_frames"])
        for c in range(K):
            centers = np.where(lab == c)[0]
            for cfr in centers:
                s = int(max(0, min(int(cfr) - fw // 2, nfr - fw)))
                pools[c].append((vi, s))
    counts = np.array([len(p) for p in pools], dtype=np.float64)

    if samples_per_class is not None and samples_per_class > 0:
        target = int(samples_per_class)
    else:
        present = counts[counts > 0]
        target = math.ceil(float(np.mean(present))) if len(present) else 1

    windows = []
    for c in range(K):
        pool = pools[c]
        n_c = len(pool)
        if n_c == 0:
            continue
        if n_c >= target:                       # 常见类: 下采样到目标
            idx = rng.choice(n_c, size=target, replace=False)
            windows += [pool[int(i)] for i in idx]
        else:                                   # 稀有类: 复制补到目标
            idx = rng.integers(0, n_c, size=target - n_c)
            windows += pool + [pool[int(i)] for i in idx]

    dataset.fixed_windows = windows
    weights = torch.ones(len(windows))
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(windows),
                                 replacement=True, generator=gen)


def build_loader(dataset, cfg, split="train", batch_size=8, num_workers=4,
                 shuffle=True, balanced=True, seed=42):
    """构造 DataLoader; 训练开启 Swin-Panda 均匀过采样 (含每类目标上限). """
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=(split == "train"))

    if split == "train" and balanced and hasattr(dataset, "videos"):
        spc = cfg["TRAIN"].get("samples_per_class")
        sampler = build_oversampled_sampler(dataset, samples_per_class=spc, seed=seed)
        kw["shuffle"] = False
        kw["sampler"] = sampler
    else:
        kw["shuffle"] = shuffle
    return DataLoader(dataset, **kw)


__all__ = ["build_loader", "build_oversampled_sampler",
           "FrameLevelDataset", "make_pos_weight_from_videos"]