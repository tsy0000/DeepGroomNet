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
from infer import load_model, predict_clip, video_frame_count
from data.clip_sampler import infer_sliding_windows, infer_syllable_clips
from postprocess.syllable_vote import apply_syllable_pipeline
from eval_metrics import classwise_and_overall, boundary_error

def labels_to_ids(labels_df, class_names):
    cols = [c for c in class_names if c in labels_df.columns]
    if len(cols) != len(class_names):
        raise ValueError(f'labels csv 缺少类别列: {set(class_names) - set(labels_df.columns)}')
    num = labels_df[cols].to_numpy().astype(float)
    s = num.sum(axis=1)
    valid = np.abs(s - 1.0) < 1e-06
    idx = np.argmax(num, axis=1)
    return (idx, valid)

def bouts_from_frames(frames_labels):
    lab = np.asarray(frames_labels)
    bouts = []
    n = len(lab)
    i = 0
    while i < n:
        j = i + 1
        while j < n and lab[j] == lab[i]:
            j += 1
        bouts.append((i, j, int(lab[i])))
        i = j
    return bouts

def run_inference(cfg, model, class_names, video, syllables, device, window_mode, crop_boxer=None):
    nf = video_frame_count(video)
    if window_mode == 'syllable':
        if syllables is None:
            raise RuntimeError("window_mode='syllable' 需要音节文件")
        clips = infer_syllable_clips(syllables, cfg)
    else:
        clips = infer_sliding_windows(nf, cfg)
    frame_logits = np.zeros((nf, len(class_names)), dtype=np.float32)
    counts = np.zeros(nf, dtype=np.float32)
    with torch.no_grad():
        for (s, e) in clips:
            log = predict_clip(model, video, s, min(e, nf), cfg, device, crop_boxer=crop_boxer).cpu().numpy()
            seg = slice(s, min(e, nf))
            frame_logits[seg] += log
            counts[seg] += 1.0
    valid = counts > 0
    frame_logits[valid] /= counts[valid, None]
    pred = frame_logits.argmax(1)
    if syllables is not None:
        (_, pred) = apply_syllable_pipeline(pred, syllables, cfg, class_names=class_names)
    else:
        from postprocess.sequence_decode import sequence_decode
        from postprocess.temporal_smoothing import smooth
        method = cfg['POSTPROCESS'].get('smoothing', 'sequence')
        if method == 'sequence':
            pred = sequence_decode(pred, cfg, class_names=class_names)
        else:
            pred = smooth(pred, method=method, window=cfg['POSTPROCESS'].get('vote_window', 5))
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--videos', nargs='*', default=None, help='video:labels.csv 对（不指定则用测试小鼠自动取）')
    ap.add_argument('--syllables', nargs='*', default=None, help='mouse:state_csv 对（可选）')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--window-mode', choices=['sliding', 'syllable'], default=None)
    ap.add_argument('--crop-mode', choices=['none', 'window', 'static'], default=None, help='ROI 裁剪（默认取 config.INFER.crop_mode=window，与训练同策略）')
    ap.add_argument('--out', default='outputs/metrics/infer_vs_annotation.csv')
    ap.add_argument('--llm-error-analysis', action='store_true', help='LLM 错误分析（模块D-2：LLM 参与验证）——从混淆矩阵+错误片段提炼行为学错误模式，输出 JSON 到 outputs/metrics/')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    class_names = list(cfg['CLASSES']['openfield_keep'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = args.ckpt or os.path.join(cfg['OUTPUT']['checkpoint_dir'], 'best_model.pt')
    (model, ckpt_class_names) = load_model(cfg, ckpt, device)
    assert list(ckpt_class_names) == class_names, f'ckpt 类顺序 {ckpt_class_names} != config {class_names}'
    window_mode = args.window_mode or cfg['INFER'].get('window_mode', 'sliding')
    if args.crop_mode is not None:
        cfg['INFER']['crop_mode'] = args.crop_mode
    from postprocess.crop_box import build_crop_boxer
    try:
        crop_boxer = build_crop_boxer(cfg)
    except FileNotFoundError as e:
        print(f'[05] 警告: {e}（回退整帧 resize）')
        crop_boxer = None
    if args.videos:
        pairs = [v.split(':') for v in args.videos]
    else:
        import train as train_mod
        meta_csv = os.path.join(cfg['DATA']['meta_root'], 'openfield_meta.csv')
        if not os.path.exists(meta_csv):
            from data.openfield_meta import build_openfield_meta
            build_openfield_meta(cfg['DATA']['openfield_roots'], meta_csv)
        meta = pd.read_csv(meta_csv)
        pseudo = [{'video': m['video'], 'start': 0, 'end': 0, 'label': 'background'} for (_, m) in meta.iterrows()]
        (_, _, te) = train_mod.split_clips(pseudo, val_ratio=cfg['TRAIN']['split']['val_ratio'], test_ratio=cfg['TRAIN']['split']['test_ratio'], seed=cfg['TRAIN']['seed'])
        test_videos = {c['video'] for c in te}
        pairs = [(m['video'], m['labels']) for (_, m) in meta.iterrows() if m['video'] in test_videos]
        print(f"[05] 测试小鼠视频 {len(pairs)} 个（划分 seed={cfg['TRAIN']['seed']}）")
    syl_map = {}
    for kv in args.syllables or []:
        (k, v) = kv.split(':')
        syl_map[k] = v
    (all_t, all_p) = ([], [])
    per_video = []
    video_gt_pred = []
    for (video, labels_csv) in pairs:
        mouse = os.path.basename(os.path.dirname(video))
        lab = pd.read_csv(labels_csv)
        lab = lab.drop(columns=[lab.columns[0]] if lab.columns[0].startswith('Unnamed') else [])
        lab = lab.rename(columns={c: 'background' for c in lab.columns if c.lower() == 'background' and c != 'background'})
        (gt, valid) = labels_to_ids(lab, class_names)
        nf = min(len(gt), video_frame_count(video))
        syllables = None
        if mouse in syl_map and os.path.exists(syl_map[mouse]):
            sdf = pd.read_csv(syl_map[mouse])
            syl = sdf[sdf.columns[0]].to_numpy()[:nf]
            if len(syl) < nf:
                syl = np.pad(syl, (0, nf - len(syl)), mode='edge')
            syllables = syl
        pred = run_inference(cfg, model, class_names, video, syllables, device, window_mode, crop_boxer=crop_boxer)
        pred = pred[:nf]
        v = valid[:nf]
        all_t.extend(gt[:nf][v].tolist())
        all_p.extend(pred[v].tolist())
        video_gt_pred.append((os.path.basename(video), gt[:nf][v].tolist(), pred[v].tolist()))
        (ov, per, cm) = classwise_and_overall(gt[:nf][v], pred[v], class_names)
        gt_bouts = bouts_from_frames(gt[:nf][v])
        pr_bouts = bouts_from_frames(pred[v])
        be = boundary_error(gt_bouts, pr_bouts)
        n_excl = int((~v).sum())
        per_video.append({'video': os.path.basename(video), 'mouse': mouse, 'acc': ov['acc'], 'f1_macro': ov['f1_macro'], 'f1_weighted': ov['f1_weighted'], 'onset_90th_pct': be['onset_90th_pct'], 'offset_90th_pct': be['offset_90th_pct']})
        print(f"[05] {mouse} acc={ov['acc']:.2f} f1m={ov['f1_macro']:.2f} onset90={be['onset_90th_pct']:.1f}帧 offset90={be['offset_90th_pct']:.1f}帧 window_mode={window_mode}" + (f'（排除非one-hot GT {n_excl} 帧）' if n_excl else ''))
    (overall, per_class, cm) = classwise_and_overall(all_t, all_p, class_names)
    print('\n[05] 汇总（帧级，含音节投票后处理）:')
    print(f"  acc={overall['acc']:.2f}  f1_macro={overall['f1_macro']:.2f}  f1_weighted={overall['f1_weighted']:.2f}")
    for (k, v) in per_class.items():
        print(f"  {k}: recall={v['recall']:.2f} P={v['precision']:.2f} F1={v['f1']:.2f}")
    yt = np.asarray(all_t, dtype=np.int64)
    yp = np.asarray(all_p, dtype=np.int64)
    print('\n[05] per-class one-vs-rest（DeepEthogram 口径）:')
    ovr = {}
    for (i, c) in enumerate(class_names):
        t = (yt == i).astype(int)
        p = (yp == i).astype(int)
        tn = int(((t == 0) & (p == 0)).sum())
        tp = int(((t == 1) & (p == 1)).sum())
        fn = int(((t == 1) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum())
        acc_ovr = (tp + tn) / max(1, tp + tn + fp + fn) * 100
        recall = tp / max(1, tp + fn) * 100
        prec = tp / max(1, tp + fp) * 100
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        ovr[c] = {'acc': acc_ovr, 'recall': recall, 'precision': prec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
        print(f'  {c}: acc={acc_ovr:.2f} recall={recall:.2f} P={prec:.2f} F1={f1:.2f} (TP={tp} FP={fp} FN={fn})')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scope', 'class', 'acc', 'precision', 'recall', 'f1', 'n'])
        w.writerow(['overall', '-', f"{overall['acc']:.2f}", f"{overall['precision_macro']:.2f}", f"{overall['recall_macro']:.2f}", f"{overall['f1_macro']:.2f}", len(all_t)])
        for (k, v) in per_class.items():
            w.writerow(['class', k, f"{v['accuracy']:.2f}", f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", v['n']])
        w.writerow([])
        w.writerow(['scope', 'class', 'acc_ovr', 'recall_ovr', 'precision_ovr', 'f1_ovr', 'tp', 'fp', 'fn'])
        for (k, v) in ovr.items():
            w.writerow(['one-vs-rest', k, f"{v['acc']:.2f}", f"{v['recall']:.2f}", f"{v['precision']:.2f}", f"{v['f1']:.2f}", v['tp'], v['fp'], v['fn']])
        w.writerow([])
        w.writerow(['video', 'mouse', 'acc', 'f1_macro', 'f1_weighted', 'onset_90th_pct_frame', 'offset_90th_pct_frame'])
        for r in per_video:
            w.writerow([r['video'], r['mouse'], f"{r['acc']:.2f}", f"{r['f1_macro']:.2f}", f"{r['f1_weighted']:.2f}", f"{r['onset_90th_pct']:.1f}", f"{r['offset_90th_pct']:.1f}"])
    cm_dir = os.path.join(os.path.dirname(args.out), 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    np.save(os.path.join(cm_dir, 'infer_vs_annotation_cm.npy'), np.asarray(cm))
    print(f'[05] saved -> {args.out}')
    if args.llm_error_analysis:
        import json
        from llm.client import build_llm_client
        from llm.error_analysis import run_error_analysis_videos
        client = build_llm_client(cfg)
        if client is None:
            print('[05] --llm-error-analysis: config.LLM.provider=none，跳过 LLM 分析（仍保存混淆矩阵统计 JSON）')
        (report, _cm) = run_error_analysis_videos(video_gt_pred, class_names, client, language=cfg['LLM'].get('language', 'zh'))
        json_path = os.path.join(os.path.dirname(args.out), 'llm_error_analysis.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[05] LLM 错误分析 saved -> {json_path} (status={report['llm_status']})")
        if report.get('llm'):
            for m in report['llm'].get('error_modes', []):
                print(f"  [错误模式] {m.get('true_class')}->{m.get('pred_class')} ({m.get('severity')}, fixable={m.get('fixable')}): {m.get('likely_cause')}")
if __name__ == '__main__':
    main()
