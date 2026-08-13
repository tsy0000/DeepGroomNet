from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2
import numpy as np
import pandas as pd
import yaml
CLASS_COLS = ['background', 'Locomote', 'Rear', 'FaceGroom', 'BodyGroom']

def audit_one_video(video, labels_csv):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    df = pd.read_csv(labels_csv)
    return (n, len(df))

def load_exceptions(csv_path):
    ex = {}
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                mouse = r.get('mouse', '').strip()
                issue = r.get('issue', '').strip()
                if mouse and issue:
                    ex[mouse, issue] = r.get('reason', '').strip()
    return ex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='outputs/meta/data_audit.csv')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    class_names = list(cfg['CLASSES']['openfield_keep'])
    seed = int(cfg['TRAIN']['seed'])
    exceptions = load_exceptions(os.path.join(cfg['DATA']['meta_root'], 'audit_exceptions.csv'))
    report_rows = []
    fails = []

    def rec(scope, item, status, detail):
        report_rows.append({'scope': scope, 'item': item, 'status': status, 'detail': str(detail)[:500]})
        if status == 'FAIL':
            fails.append(f'{scope}/{item}: {detail}')

    def rec_issue(scope, item, status, detail, issue):
        reason = exceptions.get((item, issue))
        if status == 'FAIL' and reason:
            status = 'WARN'
            detail = f'{detail}（人工已确认：{reason}）'
        rec(scope, item, status, detail)
    meta_csv = os.path.join(cfg['DATA']['meta_root'], 'openfield_meta.csv')
    if not os.path.exists(meta_csv):
        from data.openfield_meta import build_openfield_meta
        build_openfield_meta(cfg['DATA']['openfield_roots'], meta_csv)
    meta = pd.read_csv(meta_csv)
    missing = meta[~meta['video'].apply(os.path.exists)]
    missing_l = meta[~meta['labels'].apply(os.path.exists)]
    rec('meta', 'video 存在性', 'FAIL' if len(missing) else 'PASS', f"{len(meta) - len(missing)}/{len(meta)} 存在; 缺失: {missing['video'].tolist()[:5]}")
    rec('meta', 'labels 存在性', 'FAIL' if len(missing_l) else 'PASS', f'{len(meta) - len(missing_l)}/{len(meta)} 存在')
    n_misalign = 0
    for (_, m) in meta.iterrows():
        if not os.path.exists(m['video']) or not os.path.exists(m['labels']):
            continue
        (nv, nl) = audit_one_video(m['video'], m['labels'])
        diff = nl - nv
        if diff == 0:
            rec('frame_align', m['mouse'], 'PASS', f'video={nv} frames, labels={nl} rows')
        elif abs(diff) <= 2:
            rec('frame_align', m['mouse'], 'WARN', f'video={nv} frames, labels={nl} rows（差 {abs(diff)} 帧，02 已截断处理，未修改原始数据）')
        else:
            rec_issue('frame_align', m['mouse'], 'FAIL', f'video={nv} frames, labels={nl} rows（差异 >2 帧，需人工确认数据来源）', 'frame_align')
        if nv != nl:
            n_misalign += 1
        df = pd.read_csv(m['labels'])
        df = df.drop(columns=[df.columns[0]] if df.columns[0].startswith('Unnamed') else [])
        df = df.rename(columns={c: 'background' for c in df.columns if c.lower() == 'background' and c != 'background'})
        num = df[CLASS_COLS + (['Defecate'] if 'Defecate' in df.columns else [])]
        row_sum = num.sum(axis=1)
        n_all0 = int((row_sum.abs() < 1e-06).sum())
        n_overlap = int((row_sum > 1 + 1e-06).sum())
        n_bad = n_all0 + n_overlap
        frac = n_bad / max(1, len(df))
        if n_bad == 0:
            rec('onehot', m['mouse'], 'PASS', '全部 one-hot')
        elif frac < 0.01:
            rec('onehot', m['mouse'], 'WARN', f'全0行={n_all0} 重叠行={n_overlap}（{frac * 100:.2f}%，训练端已自动排除，原始数据未改动）')
        else:
            rec_issue('onehot', m['mouse'], 'FAIL', f'全0行={n_all0} 重叠行={n_overlap}（{frac * 100:.2f}% > 1%，标注质量问题）', 'onehot')
        bad_nan = int(num.isna().sum().sum())
        rec_issue('nan', m['mouse'], 'FAIL' if bad_nan else 'PASS', f'NaN 个数={bad_nan}', 'nan')
    for root in cfg['DATA']['openfield_roots']:
        if not os.path.isdir(root):
            rec('scan', root, 'FAIL', '根目录不存在')
            continue
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            files = os.listdir(p)
            n_mp4 = sum((1 for f in files if f.lower().endswith('.mp4')))
            n_lab = sum((1 for f in files if f.lower().endswith('labels.csv')))
            if n_mp4 > 0 and n_lab == 0:
                rec('scan', f'{root}/{d}', 'WARN', f'{n_mp4} 个视频但无 labels（被 find_pairs 跳过）')
            if n_mp4 == 0 and n_lab > 0:
                rec('scan', f'{root}/{d}', 'WARN', '有 labels 但无视频')
    agg = Counter()
    for (_, m) in meta.iterrows():
        if not os.path.exists(m['labels']):
            continue
        df = pd.read_csv(m['labels'])
        df = df.drop(columns=[df.columns[0]] if df.columns[0].startswith('Unnamed') else [])
        df = df.rename(columns={c: 'background' for c in df.columns if c.lower() == 'background' and c != 'background'})
        for c in CLASS_COLS + (['Defecate'] if 'Defecate' in df.columns else []):
            agg[c] += int(df[c].sum())
    total = sum(agg.values())
    rec('dist', 'overall', 'INFO', {k: f'{v} ({v / total * 100:.2f}%)' for (k, v) in agg.items()})
    clips_csv = os.path.join(cfg['DATA']['clip_output_root'], 'clips.csv')
    if not os.path.exists(clips_csv):
        rec('clips', 'clips.csv', 'FAIL', '不存在，需运行 scripts/02_gen_clips.py')
    else:
        cl = pd.read_csv(clips_csv)
        rec('clips', 'source 列', 'FAIL' if 'source' not in cl.columns else 'PASS', 'sliding clips 缺失！需运行 scripts/02_gen_clips.py --overwrite 重新生成' if 'source' not in cl.columns else f"uniform={int((cl['source'] == 'uniform').sum())}, sliding={int((cl['source'] == 'sliding').sum())}")
        clip_mice = set(cl['mouse']) if 'mouse' in cl.columns else set()
        meta_mice = set(meta['mouse'])
        rec('clips', '小鼠覆盖', 'FAIL' if clip_mice != meta_mice else 'PASS', f'{len(clip_mice)}/{len(meta_mice)}')
        missing_files = sum((1 for p in cl['clip'] if not os.path.exists(p)))
        rec('clips', 'clip 文件存在性', 'FAIL' if missing_files else 'PASS', f'缺失 {missing_files}/{len(cl)}')
        rec('clips', '标签分布', 'INFO', dict(Counter(cl['label']) if 'label' in cl.columns else {}))
    if os.path.exists(clips_csv) and 'source' in cl.columns:
        import train as train_mod
        uniform = cl[cl['source'] != 'sliding'].to_dict('records')
        (tr, va, te) = train_mod.split_clips(uniform, val_ratio=cfg['TRAIN']['split']['val_ratio'], test_ratio=cfg['TRAIN']['split']['test_ratio'], seed=seed)
        for (name, split) in (('train', tr), ('val', va), ('test', te)):
            rec('split', name, 'INFO', {k: v for (k, v) in Counter((c['label'] for c in split)).items()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['scope', 'item', 'status', 'detail'])
        w.writeheader()
        w.writerows(report_rows)
    n_fail = sum((1 for r in report_rows if r['status'] == 'FAIL'))
    n_warn = sum((1 for r in report_rows if r['status'] == 'WARN'))
    print(f'[07] audit done: FAIL={n_fail} WARN={n_warn} PASS={len(report_rows) - n_fail - n_warn} -> {args.out}')
    for r in report_rows:
        if r['status'] != 'PASS':
            print(f"  [{r['status']}] {r['scope']}/{r['item']}: {r['detail'][:200]}")
    if n_fail:
        print('\n[07] 存在 FAIL 项，训练前必须解决！')
        sys.exit(1)
if __name__ == '__main__':
    main()
