import argparse
import json
import os
import random
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2
import numpy as np
import pandas as pd
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_per_video', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, 'configs', 'config.yaml'), encoding='utf-8'))
    meta_csv = os.path.join(cfg['DATA']['meta_root'], 'openfield_meta.csv')
    if not os.path.exists(meta_csv):
        from data.openfield_meta import build_openfield_meta
        build_openfield_meta(cfg['DATA']['openfield_roots'], meta_csv)
    meta = pd.read_csv(meta_csv)
    rng = random.Random(args.seed)
    sums = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    cnt = 0
    total_frames = int(meta['total_frames'].sum())
    print(f'[00] videos={len(meta)} total_frames={total_frames}')
    for (_, m) in meta.iterrows():
        cap = cv2.VideoCapture(m['video'])
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_samp = min(args.frames_per_video, n)
        idxs = sorted(rng.sample(range(n), n_samp))
        for j in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            (ok, fr) = cap.read()
            if not ok:
                continue
            fr = cv2.resize(fr, (224, 224))
            fr = fr.astype(np.float32) / 255.0
            sums += fr.sum(axis=(0, 1))
            sq += (fr ** 2).sum(axis=(0, 1))
            cnt += 224 * 224
        cap.release()
    mean = sums / cnt
    std = np.sqrt(np.clip(sq / cnt - mean ** 2, 1e-12, None))
    stats = {'mean': mean.round(4).tolist(), 'std': std.round(4).tolist(), 'n_frames': cnt // (224 * 224), 'seed': args.seed, 'frames_per_video': args.frames_per_video}
    out = os.path.join(cfg['DATA']['meta_root'], 'video_dataset_stats.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[00] mean={stats['mean']} std={stats['std']} -> {out}")
if __name__ == '__main__':
    main()
