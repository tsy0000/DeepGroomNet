import csv
import os
from pathlib import Path
import pandas as pd

def find_pairs(root: str):
    root = Path(root)
    pairs = []
    video_dirs = sorted([d for d in root.glob('*') if d.is_dir()])
    for d in video_dirs:
        mp4s = sorted(d.glob('*.mp4'))
        labels = sorted(d.glob('*labels.csv'))
        if not mp4s or not labels:
            continue
        pairs.append({'mouse': d.name, 'video': str(mp4s[0]), 'labels': str(labels[0])})
    return pairs

def build_openfield_meta(roots, out_csv):
    rows = []
    for root in roots:
        for p in find_pairs(root):
            df = pd.read_csv(p['labels'])
            rows.append({'mouse': p['mouse'], 'video': p['video'], 'labels': p['labels'], 'total_frames': len(df)})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['mouse', 'video', 'labels', 'total_frames'])
        w.writeheader()
        w.writerows(rows)
    print(f'[openfield_meta] {len(rows)} videos -> {out_csv}')
    return rows
if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('configs/config.yaml', encoding='utf-8'))
    build_openfield_meta(cfg['DATA']['openfield_roots'], os.path.join(cfg['DATA']['meta_root'], 'openfield_meta.csv'))
