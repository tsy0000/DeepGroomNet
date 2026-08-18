"""诊断训练动力学是否正常 (不依赖最终准确率)。
回答: 梯度下降相关的指标是否在正常 ML 水平:
  1. 损失尺度 vs 随机基线损失 (模型是否比"瞎猜"好
  2. 梯度范数 (是否 NaN / 爆炸 / 消失)
  3. 参数更新幅度 (AdamW 是否在正常移动)
  4. 训练 acc vs 随机/常类基线
   5. 每类正样本预测率 (是否退化)
  6. 损失是否在下降
鐢ㄦ硶: D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/diag_train.py --steps 30 --subset-videos 6
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
from scripts.train import build_model, cosine_with_warmup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--subset-videos", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    fw = int(cfg["CLIP"]["num_input_frames"])

    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    train_v = split["train"][:args.subset_videos]
    tr = FrameLevelDataset(train_v, cfg, split="train", class_names=class_names,
                           frame_window=fw, random_start=False, clips_per_video=40)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    tr_ld = DataLoader(tr, batch_size=8, shuffle=True, num_workers=0)
    print(f"device={device} | 视频数={len(train_v)} | 采样数/epoch={len(tr)}")

    # ---- 闅忔満鍩虹嚎 ----
    pi = cc / cc.sum()
    print("\n=== 类别先验 ===")
    for i, c in enumerate(class_names):
        print(f"  {c}: prior={pi[i]*100:.2f}%  pos_weight={pw[i]:.2f}")
    chance_acc_majority = float(pi.max())
    chance_loss = 0.0
    eps = 1e-7
    # 随机模型(每类 sigmoid=prior) 的加权 BCE 期望损失(含 label smoothing)
    ls = float(cfg["LOSS"].get("binary_label_smoothing", 0.05)) * 2
    for c in range(num_classes):
        p = float(pi[c]); pwv = float(pw[c])
        sig = np.clip(p, eps, 1 - eps)
        tl = p * (1 - ls) + 0.5 * ls   # 平滑标签 (对正样本)
        tl0 = (1 - p) * (1 - ls) + 0.5 * ls  # 平滑标签 (对负样本)
        bce_pos = -(tl * math.log(sig) + (1 - tl) * math.log(1 - sig))      # 真值为正的样本
    print(f"随机基线: 常类acc(除background)={chance_acc_majority*100:.1f}% | "
          f"均匀随机acc={100/num_classes:.1f}% | 随机损失<={chance_loss:.3f}")

    # ---- 建模型 + 损失 ----
    model = build_model(cfg, num_classes).to(device)
    model.set_head(num_classes, class_counts=cc)
    model.freeze_backbone()
    opt = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad],
                            lr=5e-4, weight_decay=0.02)
    criterion = FocalLossBCE(pos_weight=pw.to(device), label_smoothing=0.05,
                                exclusive_pairs=cfg["LOSS"].get("binary_exclusivity_pairs"),
                                exclusive_weight=float(cfg["LOSS"].get("binary_exclusivity_weight", 0.0)))

    print("\n=== 前向 + 梯度诊断 ===")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数(LP): {n_params}")
    losses = []
    pos_rate = np.zeros(num_classes)   # 每类正预测率
    first_xy = None
    for step, (x, y) in enumerate(tr_ld):
        if step >= args.steps: break
        x, y = x.to(device), y.to(device)
        model.train()
        logits = model(x)
        oh = nn.functional.one_hot(y, num_classes=num_classes).float()
        loss = criterion(logits.float(), oh)
        opt.zero_grad(); loss.backward()
        # 梯度诊断
        g_norm2 = 0.0; g_abs_sum = 0.0; g_max = 0.0; n_nan = 0; n_par = 0
        for p in model.parameters():
            if p.grad is None: continue
            g = p.grad
            g_norm2 += (g.float() ** 2).sum().item()
            g_abs_sum += g.float().abs().sum().item()
            g_max = max(g_max, g.float().abs().max().item())
            n_par += g.numel()
            n_nan += int(torch.isnan(g).sum().item()) + int(torch.isinf(g).sum().item())
        grad_l2 = math.sqrt(g_norm2)
        grad_abs_mean = g_abs_sum / max(n_par, 1)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        # 预测分布
        with torch.no_grad():
            pr = torch.sigmoid(logits.float()).cpu().numpy()
            pos_rate += (pr > 0.5).sum(0)
        if first_xy is None:
            first_xy = (x.detach().cpu().numpy(), y.detach().cpu().numpy())
        if step in (0, 1, args.steps // 2, args.steps - 1):
            print(f"  step{step:3d} loss={loss.item():.3f} | grad_L2={grad_l2:.2e} "
                  f"grad_mean_abs={grad_abs_mean:.2e} grad_max={g_max:.2e} NaN={n_nan}")

    # 汇总
    loss_arr = np.array(losses)
    print(f"\n=== 汇总 ===")
    print(f"损失: 初始={loss_arr[0]:.3f} 末={loss_arr[-1]:.3f} 降幅={(loss_arr[0]-loss_arr[-1]):.3f}")
    print(f"    (随机基线<={chance_loss:.3f})  ->  实际损失{'<随机' if loss_arr[-1] < chance_loss else '~=随机'}")
    print(f"训练acc vs 随机: 见每步; 常类基线~{chance_acc_majority*100:.1f}%, 均匀~{100/num_classes:.1f}%)")
    pos_rate = pos_rate / len(losses)
    print(f"每类正预测率(>0.5): " + ", ".join(f"{c}={pr*100:.1f}%" for c, pr in zip(class_names, pos_rate)))
    print(f"    (若多数类~100% 且稀有类~0% 属正常; 若全~50% 说明模型饱和退化)")


if __name__ == "__main__":
    main()
