"""test 集正式评估：与训练完全相同的划分/口径，val 上调阈值 → test 上报告。

用途：在训练中途（last_*.pt）或完成后（best_model.pt）评估真实泛化能力。
val 只有 31 个 BG 样本（不可信），test 有 163 个——test 数字更有代表性。
"""
import sys, os, csv, json
sys.path.insert(0, r'D:\DeepGroomNet_v2')
import numpy as np
import torch
import yaml

from data.dataset import ClipDataset
from data.clip_sampler import extract_uniform_clips, extract_sliding_clips
from models.backbones.tadaconvnext import build_model
from losses.focal_loss import build_loss
from eval_metrics import classwise_and_overall, binary_classwise_metrics
from postprocess.thresholds import tune_thresholds, threshold_predict
from postprocess.sequence_decode import sequence_decode

CKPT = r"D:\DeepGroomNet_v2\outputs\checkpoints\last_openfield-s2.pt"
CONFIG = r"D:\DeepGroomNet_v2\configs\config.yaml"

cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state"] if "model_state" in ck else ck["state_dict"]
model = build_model(cfg).to(device)
r = model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
if r.missing_keys or r.unexpected_keys:
    print(f"[ckpt] missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")
model.eval()
class_names = cfg["CLASSES"]["openfield_keep"]
bg_idx = class_names.index("background")
print("ckpt epoch:", ck.get("epoch"), "best_f1:", round(ck.get("best_f1", -1), 2))

import train as T
meta_csv = os.path.join(cfg["DATA"]["meta_root"], "openfield_meta.csv")
uniform, sliding, _ = T.build_clip_rows(meta_csv, cfg)
tr, va, te = T.split_clips(uniform, val_ratio=0.15, test_ratio=0.15, seed=42)
print(f"val={len(va)} test={len(te)}")
from collections import Counter
print("val 分布:", dict(Counter(c['label'] for c in va)))
print("test 分布:", dict(Counter(c['label'] for c in te)))

def collect(model, clips, tag):
    ds = ClipDataset(clips, cfg, split="val")
    ld = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                     num_workers=0, pin_memory=False)
    ps, ys = [], []
    with torch.no_grad():
        for x, pose, kp_map, y in ld:
            x, pose, kp_map = x.to(device), pose.to(device), kp_map.to(device)
            logits = model(x, pose=pose, kp_map=kp_map).float()
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.extend(y.tolist())
    p = np.concatenate(ps); y = np.array(ys)
    print(f"[{tag}] {len(y)} clips")
    return p, y

va_p, va_y = collect(model, va, "val")
te_p, te_y = collect(model, te, "test")

from postprocess.sequence_decode import bout_length_percentiles
_tr_seqs = [[cl["label"] for _ in range(cfg["CLIP"]["orig_segment_len"])] for cl in tr]
min_bout_pc = bout_length_percentiles(_tr_seqs, class_names=class_names, percentile=1)
print("逐类最小 bout 长度（第1百分位）:", {n: int(v) for n, v in zip(class_names, min_bout_pc)})

th = tune_thresholds(va_p, va_y, bg_idx=bg_idx, candidates=100, passes=2, min_f1=0.05)
print("调优阈值:", {n: round(float(t), 3) for n, t in zip(class_names, th)})

def report(tag, probs, y, thresholds):
    pred = threshold_predict(probs, thresholds, bg_idx=bg_idx)
    pred_sm = sequence_decode(pred, cfg, class_names, min_bout_per_class=min_bout_pc)
    ovb, perb, _ = binary_classwise_metrics(y, probs, class_names, threshold=0.5)
    ov, per, _ = classwise_and_overall(y, pred_sm, class_names)
    print(f"\n===== {tag} | 阈值读出口（调优阈值 + Sakamoto滤波）5类硬分类 =====")
    print(f"acc={ov['acc']:.2f} f1_macro={ov['f1_macro']:.2f} f1_weighted={ov['f1_weighted']:.2f}")
    for k, v in per.items():
        print(f"  {k}: recall={v['recall']:.1f} P={v['precision']:.1f} F1={v['f1']:.1f} (n={v['n']})")
    if not np.array_equal(pred, pred_sm):
        changed = int((pred != pred_sm).sum())
        print(f"  [sakamoto] 滤波改动 {changed}/{len(pred)} 帧")
    print(f"===== {tag} | binary one-vs-rest @0.5（与训练 eval 同口径）=====")
    print(f"元素级acc={ovb['acc']:.2f} f1_macro={ovb['f1_macro']:.2f} f1_weighted={ovb['f1_weighted']:.2f}")
    for k, v in perb.items():
        print(f"  {k}: recall={v['recall']:.1f} P={v['precision']:.1f} F1={v['f1']:.1f} (n={v['n']})")
    return ov, per

report("val", va_p, va_y, th)
ov, per = report("TEST", te_p, te_y, th)

out_dir = cfg["OUTPUT"]["metric_dir"]
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "eval_test_intermediate.json"), "w", encoding="utf-8") as f:
    json.dump({"ckpt": CKPT, "epoch": ck.get("epoch"), "best_f1": ck.get("best_f1"),
               "thresholds": [float(t) for t in th],
               "test": {"acc": ov["acc"], "f1_macro": ov["f1_macro"],
                        "f1_weighted": ov["f1_weighted"],
                        "per_class": {k: v for k, v in per.items()}}}, f, indent=2)
print("\n已保存 ->", os.path.join(out_dir, "eval_test_intermediate.json"))
