"""opto 片段快速测试：每视频只推理前 FRAME_LIMIT 帧（默认 1500 帧≈50s），
验证跨视频/跨分布一致性，不跑全视频（省时间）。"""
import sys, os, json
sys.path.insert(0, r'D:\DeepGroomNet_v2')
import numpy as np
import pandas as pd
import torch
import yaml

from data.clip_sampler import infer_sliding_windows, video_frame_count
from infer import predict_clip
from models.backbones.tadaconvnext import build_model
from postprocess.thresholds import threshold_predict
from eval_metrics import classwise_and_overall

FRAME_LIMIT = int(os.environ.get("FRAME_LIMIT", "1500"))
CKPT = r"D:\DeepGroomNet_v2\outputs\checkpoints\last_openfield-s2.pt"
CONFIG = r"D:\DeepGroomNet_v2\configs\config.yaml"
VIDEOS = [
    r"E:\tsy\datasetsopto\mouse4121\mouse4121.mp4",
    r"E:\tsy\datasetsopto\mouse4122\mouse4122.mp4",
    r"E:\tsy\datasetsopto\mouse4125\mouse4125.mp4",
]

cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model_state"] if "model_state" in ck else ck["state_dict"]
model = build_model(cfg).to(device)
model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
model.eval()
class_names = cfg["CLASSES"]["openfield_keep"]
bg_idx = class_names.index("background")
print(f"ckpt epoch={ck.get('epoch')} best_f1={ck.get('best_f1')} | 片段帧数={FRAME_LIMIT}", flush=True)

kp_cfg = cfg.get("KEYPOINTS") or {}
bank = None
if kp_cfg.get("enabled", True) and kp_cfg.get("root"):
    from data.keypoints import KeypointBank
    bank = KeypointBank(kp_cfg["root"])
from postprocess.crop_box import build_crop_boxer
crop_boxer = None
try:
    crop_boxer = build_crop_boxer(cfg)
except FileNotFoundError as e:
    print("YOLO 回退:", e)

results = {}
for VIDEO in VIDEOS:
    mouse = os.path.basename(os.path.dirname(VIDEO))
    lab_file = os.path.join(os.path.dirname(VIDEO), f"{mouse}_labels.csv")
    kp_bodyparts = kp_array = None
    if bank is not None:
        kp_bodyparts, kp_array = bank.get(mouse)
    nf = min(video_frame_count(VIDEO), FRAME_LIMIT)
    clips = [(s, e) for (s, e) in infer_sliding_windows(nf, cfg) if s < nf]
    fl = np.zeros((nf, len(class_names)), dtype=np.float32)
    cnt = np.zeros(nf, dtype=np.float32)
    with torch.no_grad():
        for (s, e) in clips:
            log = predict_clip(model, VIDEO, s, min(e, nf), cfg, device,
                               crop_boxer=crop_boxer, kp_array=kp_array,
                               kp_bodyparts=kp_bodyparts).cpu().numpy()
            seg = slice(s, min(e, nf)); fl[seg] += log; cnt[seg] += 1.0
    v = cnt > 0; fl[v] /= cnt[v, None]
    probs = 1.0 / (1.0 + np.exp(-fl))
    pred = threshold_predict(probs, np.full(len(class_names), 0.5), bg_idx=bg_idx)
    lab = pd.read_csv(lab_file); lab.columns = [str(c).strip() for c in lab.columns]
    cols = set(lab.columns)
    n = min(len(lab), nf)
    truth = np.zeros(n, dtype=np.int64)
    for i, row in lab.iloc[:n].iterrows():
        vals = [float(row[c]) if c in cols else 0.0 for c in class_names]
        truth[i] = int(np.argmax(vals)) if abs(sum(vals) - 1.0) < 1e-6 else bg_idx
    ov, per, _ = classwise_and_overall(truth, pred[:n], class_names)
    results[mouse] = {"frames": n, "acc": ov["acc"], "f1_macro": ov["f1_macro"],
                      "per_class": {k: {kk: round(vv, 2) for kk, vv in vv.items() if kk in ("recall", "precision", "f1", "n")}
                                    for k, vv in per.items()}}
    print(f"\n===== {mouse}（片段 {n} 帧）=====", flush=True)
    print(f"acc={ov['acc']:.2f} f1_macro={ov['f1_macro']:.2f}", flush=True)
    for k, vv in per.items():
        print(f"  {k}: r={vv['recall']:.1f} P={vv['precision']:.1f} F1={vv['f1']:.1f} (n={vv['n']})", flush=True)

with open(r"D:\DeepGroomNet_v2\outputs\metrics\opto_batch_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\n已保存 outputs/metrics/opto_batch_result.json", flush=True)
