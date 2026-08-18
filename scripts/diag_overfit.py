"""鏋舵瀯璁捐鍚堢悊鎬ф祴璇曪細杩囨嫙鍚堟瀬灏忔牱鏈泦 (can-the-model-memorize)銆?
鍒ゆ柇"鏋舵瀯璁捐鏄惁鍚堢悊銆佹槸鍚﹀€煎緱鑺辨椂闂村叏閲忚缁?锛?  - 鐢?2 涓棰?鏋佸皬闆?, 璁冻澶熸鏁?  - 鑻?train loss 鏄捐憲涓嬮檷 + train acc 鍗囧埌楂?鑳借浣忓皬鏍锋湰) 鈫?鏋舵瀯鍙, 璁捐鍚堢悊, 鍊煎緱璁粌
  - 鑻?loss 鍗′綇/acc 涓婁笉鍘?杩炲皬鏍锋湰閮借繃鎷熷悎涓嶄簡) 鈫?鏋舵瀯鏈夐棶棰? 鍏堜慨鍐嶈

鍚屾椂杈撳嚭姊害鍋ュ悍搴?鏃?NaN/鐖嗙偢)銆?
鐢ㄦ硶:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_overfit.py \
      --subset-videos 2 --clip-per-video 40 --steps 120
"""
import os, sys, json, argparse, math
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

from data.dataset import FrameLevelDataset, make_pos_weight_from_videos
from losses.classification import FocalLossBCE, L2SPRegularizer
from scripts.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--subset-videos", type=int, default=2)
    ap.add_argument("--clip-per-video", type=int, default=40)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    fw = int(cfg["CLIP"]["num_input_frames"])

    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    train_v = split["train"][:args.subset_videos]
    ds = FrameLevelDataset(train_v, cfg, split="train", class_names=class_names,
                           frame_window=fw, random_start=False, clips_per_video=args.clip_per_video)
    ld = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    print(f"鏋佸皬闆? {len(train_v)} 瑙嗛, {len(ds)} 鏍锋湰, device={device}, 鍙鍙傛暟鍏ㄥ紑")

    model = build_model(cfg, num_classes).to(device)
    model.set_head(num_classes, class_counts=cc)
    model.unfreeze_all()   # 鍏ㄨВ鍐? 娴嬫灦鏋勬渶澶у涔犺兘鍔?    criterion = FocalLossBCE(pos_weight=pw.to(device), label_smoothing=0.05,
                                exclusive_pairs=cfg["LOSS"].get("binary_exclusivity_pairs"),
                                exclusive_weight=float(cfg["LOSS"].get("binary_exclusivity_weight", 0.0)))
    l2sp = L2SPRegularizer(model, alpha=0, beta=0)   # 杩囨嫙鍚堟祴璇曞叧 L2-SP, 鍙湅璁板繂鑳藉姏
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print("\n=== 杩囨嫙鍚堟祴璇?(鏋舵瀯鑳藉惁璁板繂灏忔牱鏈? ===")
    # 鎶?loader 鍏ㄩ儴 batch 鍙栧嚭, 寰幆璺戞弧 args.steps 姝?    all_batches = [(x.to(device), y.to(device)) for x, y in ld]
    print(f"batch 鏁?{len(all_batches)}, 寰幆鑷?{args.steps} 姝?)
    best_tr_acc = 0.0
    for step in range(args.steps):
        x, y = all_batches[step % len(all_batches)]
        model.train()
        out = model(x)
        logits, aux = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
        oh = nn.functional.one_hot(y, num_classes=num_classes).float()
        loss = criterion(logits.float(), oh)
        if aux is not None:
            loss = loss + float(cfg["MODEL"].get("cpam_beta", 0.1)) * criterion(aux.float(), oh)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        g_nan = sum(int(torch.isnan(p.grad).sum()) + int(torch.isinf(p.grad).sum())
                    for p in model.parameters() if p.grad is not None)
        opt.step()
        tr_acc = float((logits.detach().argmax(1) == y).float().mean())
        best_tr_acc = max(best_tr_acc, tr_acc)
        if step in (0, 1, 9, args.steps // 2, args.steps - 1):
            print(f"  step{step:3d} loss={loss.item():.3f} train_acc={tr_acc*100:.1f} NaN={g_nan}")

    print(f"\n=== 鍒ゅ畾 ===")
    print(f"鏈 loss={loss.item():.3f} | 宄板€?train_acc={best_tr_acc*100:.1f}%")
    if best_tr_acc >= 80:
        print("缁撹: 鉁?鏋舵瀯鑳借繃鎷熷悎灏忔牱鏈?鈫?璁捐鍚堢悊, 鍊煎緱鍏ㄩ噺璁粌")
    elif best_tr_acc >= 50:
        print("缁撹: 馃煛 鏋舵瀯鍙浣嗗亸寮?鈫?鍙缁? 寤鸿鍏堣皟/妫€鏌?CPAM/澶?)
    else:
        print("缁撹: 鉂?鏋舵瀯杩炲皬鏍锋湰閮借繃鎷熷悎涓嶄簡 鈫?璁捐鏈夐棶棰? 鍏堜慨鍐嶈")


if __name__ == "__main__":
    main()
