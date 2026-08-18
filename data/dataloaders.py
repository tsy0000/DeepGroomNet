# -*- coding: utf-8 -*-
"""数据加载器 + 类别不平衡处理 (过采样/加权采样, DeepEthogram 范式).

类别不平衡 (FG 4.3% / BG 1.9%) 的成熟做法 (DeepEthogram 官方):
  1. pos_weight 加权损失 (已在 losses 处理).
  2. 训练期加权采样 (oversample 稀有类) - 本模块实现.
  3. 稀有类偏置初始化 (已在 loss 内, rare_class_bias_init).

加权采样器按"每视频帧级类别频率"给样本权重: 稀有类占比高的视频权重放大,
从而每个 epoch 稀有类样本被抽到的概率提升。权重长度 = len(dataset) 精确对齐
FrameLevelDataset (每个样本 idx -> videos[idx // clips_per_video])。
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FrameLevelDataset, make_pos_weight_from_videos


def build_oversampled_sampler(dataset, power=0.5, seed=42, videos=None):
    """为 dataset 的每个样本生成采样权重, 稀有类过采样。

    weights[i]: 样本 i 所在视频的类别不平衡度。对视频内各类别频率取
    (1/freq)^power, 用视频内各类别的最大提升(稀有类占比越高权重越大),
    并归一化到与样本数对齐。

    Args:
        dataset: FrameLevelDataset (需有 .videos, .class_names, .clips_per_video)
        power: 放大指数 (DeepEthogram 用 ~0.5).
        videos: 可选, 覆盖 dataset.videos.
    """
    import numpy as np
    gen = torch.Generator().manual_seed(seed)
    videos = videos if videos is not None else dataset.videos
    class_names = dataset.class_names
    K = len(class_names)

    # 每视频: 各类别帧频率
    freq = []  # freq[v][c] = 该视频 c 类帧占比
    for v in videos:
        labels = np.asarray(v["labels"])
        cnt = np.bincount(labels, minlength=K).astype(np.float64)
        cnt = np.clip(cnt, 1e-3, None)
        total = cnt.sum()
        freq.append(cnt / total)

    n_videos = len(videos)
    cper = getattr(dataset, "clips_per_video", 1)
    n_samples = len(dataset)

    weights = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        vi = i // cper if cper > 0 else 0
        vi = min(vi, n_videos - 1)
        f = freq[vi]
        w = float(np.max((1.0 / f) ** power))   # 取该视频稀有类放大后的最大权重
        weights[i] = w

    # 归一化到均值 1, 避免数值过大
    if weights.sum() > 0:
        weights = weights / weights.mean()
    return torch.from_numpy(weights).float()


def build_loader(dataset, cfg, split="train", batch_size=8, num_workers=4,
                 shuffle=True, weighted=False, seed=42):
    """构造 DataLoader, 可选加权采样 (过采样)."""
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=(split == "train"))

    if split == "train" and weighted and hasattr(dataset, "videos"):
        weights = build_oversampled_sampler(dataset)
        gen = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True, generator=gen)
        kw["shuffle"] = False
        kw["sampler"] = sampler
    else:
        kw["shuffle"] = shuffle
    return DataLoader(dataset, **kw)


__all__ = ["build_loader", "build_oversampled_sampler",
           "FrameLevelDataset", "make_pos_weight_from_videos"]