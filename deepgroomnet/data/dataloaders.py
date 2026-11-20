import math
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from .dataset import FrameLevelDataset, make_pos_weight_from_videos

def build_oversampled_sampler(dataset, samples_per_class=None, seed=42):
    import numpy as np
    import torch
    K = len(dataset.class_names)
    videos = dataset.videos
    fw = dataset.frame_window
    rng = np.random.default_rng(seed)
    pools = [[] for _ in range(K)]
    for (vi, v) in enumerate(videos):
        lab = np.asarray(v['labels'])
        nfr = int(v['n_frames'])
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
        if n_c >= target:
            idx = rng.choice(n_c, size=target, replace=False)
            windows += [pool[int(i)] for i in idx]
        else:
            idx = rng.integers(0, n_c, size=target - n_c)
            windows += pool + [pool[int(i)] for i in idx]
    dataset.fixed_windows = windows
    weights = torch.ones(len(windows))
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(windows), replacement=True, generator=gen)

def build_loader(dataset, cfg, split='train', batch_size=8, num_workers=4, shuffle=True, balanced=True, seed=42):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=split == 'train')
    if split == 'train' and balanced and hasattr(dataset, 'videos'):
        spc = cfg['TRAIN'].get('samples_per_class')
        sampler = build_oversampled_sampler(dataset, samples_per_class=spc, seed=seed)
        kw['shuffle'] = False
        kw['sampler'] = sampler
    else:
        kw['shuffle'] = shuffle
    return DataLoader(dataset, **kw)
__all__ = ['build_loader', 'build_oversampled_sampler', 'FrameLevelDataset', 'make_pos_weight_from_videos']
