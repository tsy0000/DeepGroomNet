"""训练前特征可分性诊断（不训练，用冻结预训练骨干提特征，看能否区分类别）。

成熟方法:
  - kNN probe (VideoGLUE): 冻结特征上做最近邻分类, 给"特征可分性"准确率
  - silhouette / Fisher 判别: 每类质心 + 类间/类内距离, 量化类别分离度
  - 重点看 FG(FaceGroom) vs BG(BodyGroom) 是否可分 (极细粒度)

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_separability.py \
      --subset-videos 10 --clip-per-video 40 --max-samples 2000
"""
import os, sys, json, argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

from data.dataset import FrameLevelDataset
from scripts.train_resume import build_model


def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            logits = out
            # 用预激活特征(骨干输出)比 logits 更能反映原始可分性
            feats.append(logits.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def separability_metrics(F, Y, class_names):
    """质心 + Fisher + silhouette + kNN。F: [N,D], Y: [N] int。"""
    K = len(class_names)
    N = F.shape[0]
    Fn = (F - F.mean(0)) / (F.std(0) + 1e-8)
    centroids = np.array([Fn[Y == c].mean(0) for c in range(K)])
    # Fisher: 类间散度 / 类内散度 (对角近似)
    sw = np.array([((Fn[Y == c] - centroids[c]) ** 2).sum(0) / max(1, (Y == c).sum()) for c in range(K)])
    sb = np.array([((centroids[c] - Fn.mean(0)) ** 2).sum() for c in range(K)])
    fisher = float((sb.sum() + 1e-8) / (sw.sum() + 1e-8))
    # 质心距离矩阵 (FG/BG 特意看)
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    # silhouette (简化, 用欧氏距离)
    sil = []
    for c in range(K):
        idx = np.where(Y == c)[0]
        if len(idx) < 2:
            sil.append(0.0); continue
        a = np.mean([np.linalg.norm(Fn[i] - Fn[j]) for j in idx for i in idx if i != j]) / 2
        b = min(np.mean([np.linalg.norm(Fn[i] - Fn[j]) for j in np.where(Y != c)[0]])
                for i in idx) if (Y != c).any() else 0.0
        sil.append((b - a) / max(a, b, 1e-8))
    # kNN (leave-one-out 采样子集, 效率)
    rng = np.random.RandomState(0)
    if N > 1500:
        pick = rng.choice(N, 1500, replace=False)
        Fk, Yk = Fn[pick], Y[pick]
    else:
        Fk, Yk = Fn, Y
    d = ((Fk[:, None, :] - Fk[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    knn_acc = float((Yk[np.argmin(d, 1)] == Yk).mean())
    return {"fisher": fisher, "silhouette": float(np.mean(sil)),
            "knn_acc": knn_acc, "centroid_dist": dist, "sil_per_class": sil}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--subset-videos", type=int, default=10)
    ap.add_argument("--clip-per-video", type=int, default=40)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    fw = int(cfg["CLIP"]["num_input_frames"])

    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    videos = split[args.split][:args.subset_videos]
    print(f"提取特征: {len(videos)} 视频, device={device}")

    ds = FrameLevelDataset(videos, cfg, split="train", class_names=class_names,
                           frame_window=fw, random_start=False, clips_per_video=args.clip_per_video)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model = build_model(cfg, len(class_names)).to(device)   # 冻结预训练骨干
    model.eval()
    F, Y = extract_features(model, ld, device)
    # 每类均衡采样
    rng = np.random.RandomState(42)
    keep = []
    for c in range(len(class_names)):
        idx = np.where(Y == c)[0]
        n = min(len(idx), args.max_samples // len(class_names))
        keep.extend(rng.choice(idx, n, replace=False) if n else [])
    F, Y = F[keep], Y[keep]
    print(f"样本: {len(F)} (每类≈{args.max_samples//len(class_names)})")

    m = separability_metrics(F, Y, class_names)
    print("\n=== 冻结预训练骨干的特征可分性 (未训练) ===")
    print(f"Fisher 判别比 (越大越可分): {m['fisher']:.3f}")
    print(f"kNN 准确率 (零训练): {m['knn_acc']*100:.1f}%  (随机=20%, 常类={100*max(np.bincount(Y))/len(Y):.0f}%)")
    print(f"平均 silhouette (>-0.25差, 0~0.5中, >0.5好): {m['silhouette']:.3f}")
    print(f"各类 silhouette: " + ", ".join(f"{c}={s:.2f}" for c, s in zip(class_names, m['sil_per_class'])))
    print("\n质心欧氏距离矩阵:")
    d = m["centroid_dist"]
    hdr = "           " + "".join(f"{c[:4]:>8}" for c in class_names)
    print(hdr)
    for i, c in enumerate(class_names):
        print(f"{c[:4]:>8}   " + "".join(f"{d[i,j]:>8.2f}" for j in range(len(class_names))))
    fg, bg = class_names.index("FaceGroom"), class_names.index("BodyGroom")
    print(f"\n结论: FG/BG 质心距离 = {d[fg,bg]:.2f} "
          f"({'偏小, 细粒度难分 → 需训练/CPAM' if d[fg,bg] < d.mean() else '相对可分'})")


if __name__ == "__main__":
    main()
