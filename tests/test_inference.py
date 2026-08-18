"""推理模块测试: TAdaConvModel 滑窗逐帧 + 后处理 (合成数据)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import cv2
import tempfile

from models.tadaconv_model import TAdaConvModel
from inference.predict import load_tadaconv_model, sliding_window_probs, predict_video

CFG = {
    "MODEL": {"cpam": True, "stsa": True},
    "CLIP": {"num_input_frames": 32, "target_size": 64,
             "norm_mean": 0.45, "norm_std": 0.225},
    "CLASSES": {"openfield": ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]},
    "POSTPROCESS": {"temporal_smooth": True, "smooth_window": 3,
                    "min_bout_per_behavior": True, "min_bout_len": {}},
}


def make_video(path, n_frames=64, size=64):
    """合成视频 (随机噪声帧)."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (size, size))
    for _ in range(n_frames):
        fr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
        vw.write(fr)
    vw.release()


def test_sliding_window():
    model = TAdaConvModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=True)
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        vpath = os.path.join(tmp, "t.mp4")
        make_video(vpath, n_frames=64)
        probs, cnt = sliding_window_probs(model, vpath, CFG, "cpu", stride=1)
        assert probs.shape == (64, 5), f"probs: {probs.shape}"
        assert np.all(probs >= 0) and np.all(probs <= 1)
        print(f"[OK] 滑窗: probs={probs.shape} 覆盖数={cnt.min()}-{cnt.max()}")


def test_predict_video():
    model = TAdaConvModel(num_classes=5, t_stride=1, use_cpam=True, use_stsa=True)
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        vpath = os.path.join(tmp, "t.mp4")
        make_video(vpath, n_frames=64)
        probs, pred = predict_video(model, vpath, CFG, "cpu", syllable=np.random.randint(0, 3, 64))
        assert probs.shape == (64, 5)
        assert pred.shape == (64, 5)
        assert set(np.unique(pred)) <= {0, 1}
        print(f"[OK] 推理+后处理: probs={probs.shape} pred={pred.shape}")


def test_load_model():
    with tempfile.TemporaryDirectory() as tmp:
        # 构造一个权重文件
        m = TAdaConvModel(num_classes=5, t_stride=1)
        ck = os.path.join(tmp, "m.pt")
        torch.save({"model_state": m.state_dict()}, ck)
        m2 = load_tadaconv_model(CFG, 5, ckpt_path=ck, device="cpu")
        assert m2 is not None
        print("[OK] 模型加载正常")


if __name__ == "__main__":
    test_sliding_window()
    test_predict_video()
    test_load_model()
    print("推理模块全部测试通过")