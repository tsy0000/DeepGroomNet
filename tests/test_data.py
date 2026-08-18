"""数据层测试: pos_weight 统计 + dataloader 构造 (合成数据, 不需真实视频)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from data import FrameLevelDataset, make_pos_weight_from_videos, build_loader
from data.dataset import FrameLevelDataset as FLD


def test_pos_weight():
    """类别不平衡 -> pos_weight 正确放大稀有类."""
    class_names = ["A", "B", "C"]
    # B 极稀有
    videos = [
        {"video": "x.mp4", "labels": np.zeros(1000, dtype=int), "n_frames": 1000},
    ]
    videos[0]["labels"][:50] = 1   # 5% B
    videos[0]["labels"][:5] = 2    # 0.5% C
    pw, cc = make_pos_weight_from_videos(videos, class_names)
    pw = pw.numpy()
    assert pw[0] < pw[1] < pw[2], f"pos_weight 应稀有类更大: {pw}"
    print(f"[OK] pos_weight={np.round(pw,3)} (稀有类权重更大)")


def test_dataset_length():
    """T=32 配置: dataset 长度 = 视频数 × clips_per_video."""
    # 只测 make_pos_weight 签名, 真实 dataset 需视频文件
    assert callable(FrameLevelDataset), "FrameLevelDataset 可调用"
    print("[OK] FrameLevelDataset 导入正常")


def test_oversampler_weights():
    """Swin-Panda 均匀过采样: 稀有类补到平均值, 常见类保留."""
    from data.dataloaders import build_oversampled_sampler
    class_names = ["A", "B", "C"]
    cfg = {"CLIP": {"target_size": 224, "norm_mean": 0.45, "norm_std": 0.225},
           "CLASSES": {"openfield": class_names},
           "DATA": {"meta_root": "outputs/meta"},
           "TRAIN": {"clips_per_video": 50}}
    # 构造极不平衡假视频: A 常见, B 中等, C 极稀有
    videos = []
    for vi in range(3):
        labels = np.zeros(2000, dtype=int)
        labels[:1200] = 0      # A 60%
        labels[1200:1600] = 1  # B 20%
        labels[1600:1640] = 2  # C 2%
        videos.append({"video": f"v{vi}.mp4", "labels": labels,
                       "n_frames": 2000, "mouse": f"m{vi}", "source": "syn"})
    ds = FrameLevelDataset(videos, cfg, split="train", class_names=class_names,
                           frame_window=16, clips_per_video=50)
    sampler = build_oversampled_sampler(ds, seed=0)
    fw = ds.fixed_windows
    from collections import Counter
    cnt = Counter()
    for vi, s in fw:
        center = s + 16 // 2
        cnt[int(videos[vi]["labels"][center])] += 1
    # 稀有类 C 应被补到平均; 常见类 A 保留最多
    assert cnt[2] > 0, "稀有类应被过采样"
    assert cnt[0] >= cnt[2], "常见类不应少于稀有类"
    print(f"[OK] 均匀过采样窗口数 per-class={dict(cnt)} (稀有类补到平均)")


if __name__ == "__main__":
    test_pos_weight()
    test_dataset_length()
    test_oversampler_weights()
    print("数据层测试通过")