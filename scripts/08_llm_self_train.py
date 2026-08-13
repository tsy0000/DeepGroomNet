from __future__ import annotations
import argparse
import csv
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import pandas as pd
import torch
import yaml
from infer import load_model, video_frame_count
from data.clip_sampler import infer_sliding_windows
from postprocess.crop_box import build_crop_boxer
from llm.client import build_llm_client
from llm.pseudo_label import select_pseudo_labels
PSEUDO_CSV = os.path.join(ROOT, 'outputs', 'clips', 'clips_pseudo.csv')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--video', required=True, help='未标注视频（自训练数据源）')
    ap.add_argument('--syllables', default=None, help='可选 MoSeq 音节文件')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--min-conf', type=float, default=None)
    ap.add_argument('--min-len', type=int, default=None)
    ap.add_argument('--max-samples', type=int, default=None)
    ap.add_argument('--target', nargs='*', default=None, help='伪标签目标类（默认 FaceGroom BodyGroom）')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    st = cfg.get('SELFTRAIN', {})
    if not st.get('enabled', False) and args.min_conf is None:
        print('[08] 注意：config.SELFTRAIN.enabled=false（默认）。本脚本单次运行不受影响，但全流程自训练请开启后重训。')
    min_conf = args.min_conf if args.min_conf is not None else float(st.get('min_conf', 0.85))
    min_len = args.min_len if args.min_len is not None else int(st.get('min_len', 8))
    max_samples = args.max_samples if args.max_samples is not None else int(st.get('max_samples', 200))
    target = args.target or list(st.get('target_classes', ['FaceGroom', 'BodyGroom']))
    language = cfg.get('LLM', {}).get('language', 'zh')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or os.path.join(cfg['OUTPUT']['checkpoint_dir'], 'best_model.pt')
    (model, class_names) = load_model(cfg, ckpt, device)
    client = build_llm_client(cfg)
    if client is None:
        print('[08] 警告：LLM 未配置（provider=none），伪标签仅经规则校验（置信度阈值），建议配置 LLM（config.LLM）做语义校验。')
    nf = video_frame_count(args.video)
    syllables = None
    if args.syllables and os.path.exists(args.syllables):
        df = pd.read_csv(args.syllables)
        syl = df[df.columns[0]].to_numpy()[:nf]
        if len(syl) < nf:
            syl = np.pad(syl, (0, nf - len(syl)), mode='edge')
        syllables = syl
    clips = infer_sliding_windows(nf, cfg)
    frame_logits = np.zeros((nf, len(class_names)), dtype=np.float32)
    counts = np.zeros(nf, dtype=np.float32)
    try:
        crop_boxer = build_crop_boxer(cfg)
    except FileNotFoundError:
        crop_boxer = None
    from infer import predict_clip
    with torch.no_grad():
        for (s, e) in clips:
            log = predict_clip(model, args.video, s, min(e, nf), cfg, device, crop_boxer=crop_boxer).cpu().numpy()
            seg = slice(s, min(e, nf))
            frame_logits[seg] += log
            counts[seg] += 1.0
    valid = counts > 0
    frame_logits[valid] /= counts[valid, None]
    pred = frame_logits.argmax(1)
    conf = frame_logits.max(1)
    if syllables is not None:
        from postprocess.syllable_vote import apply_syllable_pipeline
        (_, pred) = apply_syllable_pipeline(pred, syllables, cfg, class_names=class_names)
    accepted = select_pseudo_labels(pred, conf, class_names, target, client=client, min_conf=min_conf, min_len=min_len, max_samples=max_samples, language=language)
    print(f'[08] 候选(高置信) -> LLM 校验通过: {len(accepted)} 个伪标签 bout (视频 {os.path.basename(args.video)})')
    for (s, e, c, cf) in accepted[:20]:
        print(f'   [{s}-{e}] {c} conf={cf:.3f} ({e - s + 1}帧)')
    rows = []
    for (s, e, c, cf) in accepted:
        rows.append({'video': args.video, 'start': s, 'end': e + 1, 'label': c, 'conf': round(cf, 4), 'source': 'pseudo', 'frame_count': e - s + 1})
    os.makedirs(os.path.dirname(PSEUDO_CSV), exist_ok=True)
    existing = []
    if os.path.exists(PSEUDO_CSV):
        with open(PSEUDO_CSV, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                existing.append(r)
    merged = existing + rows
    with open(PSEUDO_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['video', 'start', 'end', 'label', 'conf', 'source', 'frame_count'])
        w.writeheader()
        w.writerows(merged)
    print(f'[08] 伪标签清单已写入 {PSEUDO_CSV}（累计 {len(merged)} 条）')
    print('[08] 重训（伪标签自动并入训练集，不进 val/test）：python scripts/03_train.py')
if __name__ == '__main__':
    main()
