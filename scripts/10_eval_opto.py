import sys, os, csv, json
sys.path.insert(0, 'D:\\DeepGroomNet_v2')
import numpy as np
import pandas as pd
import torch
import yaml
from data.clip_sampler import infer_sliding_windows, video_frame_count
from infer import predict_clip
from models.backbones.tadaconvnext import build_model

def json_load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)
VIDEO = 'E:\\tsy\\datasetssap3\\mouse021\\mouse021.mp4'
LABELS = 'E:\\tsy\\datasetssap3\\mouse021\\mouse021_labels.csv'
CKPT = 'D:\\DeepGroomNet_v2\\outputs\\checkpoints\\last_openfield-s2.pt'
CONFIG = 'D:\\DeepGroomNet_v2\\configs\\config.yaml'
cfg = yaml.safe_load(open(CONFIG, encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ck = torch.load(CKPT, map_location=device, weights_only=False)
sd = ck['model_state'] if 'model_state' in ck else ck['state_dict']
model = build_model(cfg).to(device)
r = model.load_state_dict({k: v.to(device) for (k, v) in sd.items()}, strict=False)
if r.missing_keys or r.unexpected_keys:
    print(f'[ckpt] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}')
model.eval()
class_names = cfg['CLASSES']['openfield_keep']
print('classes:', class_names)
print('loaded ckpt:', CKPT, 'epoch:', ck.get('epoch'))
kp_array = kp_bodyparts = None
kp_cfg = cfg.get('KEYPOINTS') or {}
if kp_cfg.get('enabled', True) and kp_cfg.get('root'):
    from data.keypoints import KeypointBank
    bank = KeypointBank(kp_cfg['root'])
    (kp_bodyparts, kp_array) = bank.get('mouse021')
    print('keypoints:', '有' if kp_array is not None else '无')
nf = video_frame_count(VIDEO)
clips = infer_sliding_windows(nf, cfg)
print(f'video frames={nf} windows={len(clips)}')
crop_boxer = None
try:
    from postprocess.crop_box import build_crop_boxer
    crop_boxer = build_crop_boxer(cfg)
    print('YOLO 裁剪: 启用')
except FileNotFoundError as e:
    print('YOLO 裁剪: 回退整帧', e)
frame_logits = np.zeros((nf, len(class_names)), dtype=np.float32)
counts = np.zeros(nf, dtype=np.float32)
with torch.no_grad():
    for (i, (s, e)) in enumerate(clips):
        log = predict_clip(model, VIDEO, s, min(e, nf), cfg, device, crop_boxer=crop_boxer, kp_array=kp_array, kp_bodyparts=kp_bodyparts).cpu().numpy()
        seg = slice(s, min(e, nf))
        frame_logits[seg] += log
        counts[seg] += 1.0
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(clips)} 窗')
valid = counts > 0
frame_logits[valid] /= counts[valid, None]
probs = 1.0 / (1.0 + np.exp(-frame_logits))
thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
print('阈值: 0.5（与训练 eval_epoch 的 binary_classwise_metrics 口径一致）')
from postprocess.thresholds import threshold_predict
bg_idx = class_names.index('background')
pred = threshold_predict(probs, thresholds, bg_idx=bg_idx)
print('阈值读出口完成')
from postprocess.sequence_decode import sakamoto_bout_filter
pred_sak = sakamoto_bout_filter(pred, class_names=class_names)
print('Sakamoto 滤波完成')
lab = pd.read_csv(LABELS)
lab.columns = [str(c).strip() for c in lab.columns]
cols = set(lab.columns)
truth = np.zeros(len(lab), dtype=np.int64)
n_other = 0
for (i, row) in lab.iterrows():
    vals = [float(row[c]) if c in cols else 0.0 for c in class_names]
    if abs(sum(vals) - 1.0) < 1e-06:
        truth[i] = int(np.argmax(vals))
    else:
        truth[i] = bg_idx
        n_other += 1
print(f'标签帧数={len(lab)} 非标准one-hot帧(按bg计)={n_other}')
n = min(len(truth), nf)
from eval_metrics import classwise_and_overall, binary_classwise_metrics
for (name, p) in (('阈值读出口', pred[:n]), ('Sakamoto滤波后', pred_sak[:n])):
    probs_n = probs[:n]
    (ovb, perb, _) = binary_classwise_metrics(truth[:n], probs_n, class_names, threshold=0.5)
    print(f'\n===== {name} | 口径1: binary one-vs-rest（与训练 val_f1m 同口径）=====')
    print(f"元素级acc={ovb['acc']:.2f} f1_macro={ovb['f1_macro']:.2f} f1_weighted={ovb['f1_weighted']:.2f}")
    for (k, v) in perb.items():
        print(f"  {k}: recall={v['recall']:.1f} P={v['precision']:.1f} F1={v['f1']:.1f} (n={v['n']})")
    (ov, per, _) = classwise_and_overall(truth[:n], p, class_names)
    print(f'===== {name} | 口径2: 5类硬分类混淆 =====')
    print(f"整体 acc={ov['acc']:.2f} f1_macro={ov['f1_macro']:.2f} f1_weighted={ov['f1_weighted']:.2f}")
    for (k, v) in per.items():
        print(f"  {k}: recall={v['recall']:.1f} P={v['precision']:.1f} F1={v['f1']:.1f} (n={v['n']})")
