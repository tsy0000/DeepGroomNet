"""数据加载器 + 类别不平衡处理 (过采样/加权采样, DeepEthogram 范式).

类别不平衡 (FG 4.3% / BG 1.9%) 的成熟做法:
  1. pos_weight 加权损失 (已在 losses 处理).
  2. 训练期加权采样 (oversample 稀有类) - 本模块实现.

Sakamoto 每 epoch 配额思路: 用 WeightedRandomSampler 按类别频率倒数采样,
保证每个 batch 稀有类出现概率提升.
"""
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FrameLevelDataset, make_pos_weight_from_videos


def build_oversampled_sampler(videos, class_names, power=0.5, seed=42):
    """为每个视频窗口起点生成采样权重, 稀有类过采样.

    权重 = (1 / 类别频率) ^ power。每视频每 clip 起点一个样本权重。
    """
    from collections import Counter
    import random
    random.seed(seed)

    weights = []
    for v in videos:
        labels = torch.tensor(v["labels"], dtype=torch.long)
        cls = labels.float().mean(dim=0) if labels.dim() > 1 else None
        # 单标签模式: labels 是 (N,) 类索引
        n = v["n_frames"]
        start_centers = range(0, max(1, n), max(1, n // 500))  # 每视频采样中心
        for c in start_centers:
            if labels.dim() == 1:
                c_ = labels[min(c, n - 1)]
                # 稀有类放大
                freq = (labels == c_).float().mean().clamp(min=1e-3)
                w = (1.0 / freq) ** power
            else:
                # 多标签: 用最小类概率放大
                idx = min(c, n - 1)
                probs = labels[idx]
                freq = probs.mean().clamp(min=1e-3)
                w = (1.0 / freq) ** power
            weights.append(float(w))
    return weights


def build_loader(dataset, cfg, split="train", batch_size=8, num_workers=4,
                 shuffle=True, weighted=False, seed=42):
    """构造 DataLoader, 可选加权采样 (过采样)."""
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=(split == "train"))

    if split == "train" and weighted and hasattr(dataset, "videos"):
        weights = build_oversampled_sampler(dataset.videos, dataset.class_names)
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