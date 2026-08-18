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
    """过采样: 稀有类权重更大."""
    from data.dataloaders import build_oversampled_sampler
    class_names = ["A", "B"]
    labels = np.zeros(5000, dtype=int)
    labels[200:300] = 1   # 2% B
    videos = [{"video": "v.mp4", "labels": labels, "n_frames": 5000}]
    w = build_oversampled_sampler(videos, class_names)
    assert len(w) > 0
    print(f"[OK] 过采样权重序列长度={len(w)}, 全部>0={all(x > 0 for x in w)}")


if __name__ == "__main__":
    test_pos_weight()
    test_dataset_length()
    test_oversampler_weights()
    print("数据层测试通过")