"""架构设计合理性测试: 过拟合极小样本集 (can-the-model-memorize)。
判断"架构设计是否合理、是否值得花时间全量训练"。  - 用 2 个视频, 极小数据, 训足够步数;  - 若 train loss 显著下降 +
  - 若 loss 卡住/acc 上不去 (连小样本都过拟合不了) -> 结构有问题, 先修再训

同时输出梯度健康度 (无 NaN/爆炸)。
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
    print(f"极小集: {len(train_v)} 视频, {len(ds)} 样本, device={device}, 可训参数全开")

    model = build_model(cfg, num_classes).to(device)
    model.set_head(num_classes, class_counts=cc)
    model.unfreeze_all()   # 全解冻, 测架构最大学习能力
    criterion = FocalLossBCE(pos_weight=pw.to(device), label_smoothing=0.05,
                                exclusive_weight=float(cfg["LOSS"].get("binary_exclusivity_weight", 0.0)))
    l2sp = L2SPRegularizer(model, alpha=0, beta=0)   # 过拟合测试关 L2-SP, 只看记忆能力
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print("\n=== 过拟合测试 (结构能否记忆小样本) ===")
    # 抽出 loader 全部 batch, 循环跑满 args.steps 步
    all_batches = [(x.to(device), y.to(device)) for x, y in ld]
    print(f"batch 数={len(all_batches)}, 循环至 {args.steps} 步")
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

    print(f"\n=== 判定 ===")
    print(f"末步 loss={loss.item():.3f} | 峰值 train_acc={best_tr_acc*100:.1f}%")
    if best_tr_acc >= 80:
        print("结论: 结构能过拟合小样本 -> 设计合理, 值得全量训练")
    elif best_tr_acc >= 50:
        print("结论: 结构可学但偏置, 可训练; 建议先调/检查 CPAM/头)")
    else:
        print("结论: 结构连小样本都过拟合不了 -> 设计有问题, 先修再训")


if __name__ == "__main__":
    main()
