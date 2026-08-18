"""DeepGroomNet_v3 - 训练主脚本。
对齐 DeepEthogram (帧级行为分类) + TAdaConv 官方训练配置:
  - 优化器: AdamW, lr=2.5e-4, weight_decay 小 (TAdaConv 官方)
  - 分层 lr: head/新层 lr*10 (TAdaConv HEAD_LRMULT=10, 固定倍数非中途改)
  - cosine lr + warmup (TAdaConv 瀹樻柟 8ep warmup)
  - 损失: pos_weight BCE + label smoothing + L2-SP (loss 内)
  - 无配额采样器 (类别不平衡靠 pos_weight)
  - 无每 epoch 回滚 (权重单调累积, 标准做法)
  - 按小鼠划分, 实时帧级解码
"""
import os
import sys
import json
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import math
import torch
import torch.nn as nn
import yaml

from data.dataset import FrameLevelDataset, make_pos_weight_from_videos
from data.dataloaders import build_loader
from models.tadaconv_model import TAdaConvModel, load_pretrained
from losses.auxiliary import AuxFocalLoss
from postprocess.thresholds import (evaluate_thresholds, threshold_predict,
                                    compute_background, min_bout_per_behavior,
                                    binary_metrics, smooth_probabilities,
                                    hmm_smooth, bout_stats)
from torch.utils.data import DataLoader  # noqa: F401
from tqdm import tqdm


def binary_class_metrics(eye, pred, class_names):
    """逐类二值指标 (eye/pred 均为 [N,K] 0/1)。返回 per_class: acc/precision/recall/f1。"""
    per_class = {}
    N = max(1, eye.shape[0])
    for i, c in enumerate(class_names):
        t = eye[:, i]; pr = pred[:, i]
        tp = int(((t == 1) & (pr == 1)).sum())
        tn = int(((t == 0) & (pr == 0)).sum())
        fn = int(((t == 1) & (pr == 0)).sum())
        fp = int(((t == 0) & (pr == 1)).sum())
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
        per_class[c] = {"accuracy": (tp + tn) / N * 100,
                        "precision": pre * 100, "recall": rec * 100, "f1": f1 * 100}
    return per_class


def f1_macro_from(per_class, class_names):
    return float(np.mean([per_class[c]["f1"] for c in class_names]))


def macro_auroc(P, Y, class_names):
    """阈值无关的 AUROC (迁移自 DeepEthogram metrics.roc_auc / sklearn)。
    用于早停/选模型, 避免在极小 val + 稀有类上做阈值网格搜索导致 F1 抖动。    P: [N,K] 概率; Y: [N] int 标签。    """
    from sklearn.metrics import roc_auc_score
    K = P.shape[1]
    eye = (Y[:, None] == np.arange(K)).astype(int)
    scores = []
    for i in range(K):
        try:
            s = roc_auc_score(eye[:, i], P[:, i])
            if np.isfinite(s):
                scores.append(s)
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def macro_auprc(P, Y, class_names):
    """阈值无关的 AUPRC (对极不平衡更稳)。"""
    from sklearn.metrics import average_precision_score
    K = P.shape[1]
    eye = (Y[:, None] == np.arange(K)).astype(int)
    scores = []
    for i in range(K):
        try:
            s = average_precision_score(eye[:, i], P[:, i])
            if np.isfinite(s):
                scores.append(s)
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else 0.0


def build_model(cfg, num_classes):
    m = cfg["MODEL"]
    model = TAdaConvModel(
        num_classes=num_classes,
        t_stride=1,                       # T=32
        drop_path=float(m.get("drop_path", 0.2)),
        use_cpam=bool(m.get("cpam", True)),
        use_stsa=bool(m.get("stsa", True)),
    )
    w = m.get("pretrained")
    if w and os.path.exists(w):
        load_pretrained(model, w)
    return model


def build_videos_from_root(root, source, class_names=None):
    """扫描一个目录的所有小鼠视频, 构建测试用 videos 清单 (对齐 02_build_videos)。

    标签列对齐 5 类 (Locomote/Rear/FaceGroom/BodyGroom/background):
    未识别列 (如 Defecate/Jump) 不纳入, 帧若仅含这些行为则归 background。
    """
    import pandas as pd
    import cv2 as _cv2
    class_names = class_names or ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]
    bg_idx = len(class_names) - 1
    idx_map = {c: i for i, c in enumerate(class_names)}
    videos = []
    if not os.path.isdir(root):
        print(f"  跳过不存在: {root}")
        return videos
    for mouse in sorted(os.listdir(root)):
        mdir = os.path.join(root, mouse)
        if not os.path.isdir(mdir):
            continue
        label_csv, video_mp4 = None, None
        for f in os.listdir(mdir):
            if f.endswith(".csv") and ("label" in f.lower() or f.startswith("labels") or f.lower() == f"{mouse.lower()}.csv"):
                if label_csv is None or "labels2" in f:
                    label_csv = os.path.join(mdir, f)
            elif f.endswith(".mp4"):
                video_mp4 = os.path.join(mdir, f)
        if not (label_csv and video_mp4):
            print(f"  跳过 {mouse}: 缺标签/视频")
            continue
        try:
            df = pd.read_csv(label_csv)
            labels = np.full(len(df), bg_idx, dtype=np.int64)
            for i in range(len(df)):
                active = [idx_map[c] for c in class_names[:-1]
                          if c in df.columns and df[c].iloc[i] == 1]
                labels[i] = active[0] if active else bg_idx
            cap = _cv2.VideoCapture(video_mp4)
            n = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if len(labels) > n:
                labels = labels[:n]
        except Exception as e:
            print(f"  跳过 {mouse}: {e}")
            continue
        videos.append({"video": video_mp4, "mouse": mouse, "source": source,
                       "n_frames": len(labels), "labels": labels.tolist()})
        print(f"  {mouse}: {len(labels)} 帧")
    return videos


def build_oft_videos(root, class_names=None, fps=25.0, labeler="Jin"):
    """Sturman-OFT 预训练数据: bout区间标签 (from/to秒 + type) -> 逐帧类索引。

    OFT 标签文件 AllLabDataOFT_final.csv 含 3 位标注者, 每行一条 bout:
      file;Experimenter;from(s);to(s);type;ID;DLCFile
    类型: Supported/Unsupported/Grooming/Jumping/StartEnd/_DEFAULT/NA。
    background = 无任何行为。每个视频按 ID 单独构建。
    Args:
        root: OFT 数据目录 (含 AllLabDataOFT_final.csv + OFT_*.mp4)
        class_names: [background, Supported, Unsupported, Grooming, Jumping]
        fps: OFT 视频帧率 (Sturman 为 25fps)。
        labeler: 标注者名 ('Jin'/'Furkan'/'Oliver') 或 'ALL'(合并3人)。
            论文图2H 用 Jin (论文方法: "chose Jin as the labeler"),
            类别占比与论文一致需用单标注者 Jin。
    """
    import cv2 as _cv2
    import numpy as _np
    from collections import Counter
    offtype_to_idx = {c: i for i, c in enumerate(class_names)}
    bg = 0
    videos = []
    if not os.path.isdir(root):
        print(f"  OFT 根目录不存在: {root}")
        return videos
    csv_path = None
    for f in sorted(os.listdir(root)):
        if f.endswith(".csv"):
            csv_path = os.path.join(root, f)
            break
    if not csv_path:
        print("  OFT: 未找到标注CSV")
        return videos
    # 读 bout 标注, 记录每个视频 ID + 标注者
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    per_video = {}  # vid_id -> list of (fr, to, idx)
    for ln in raw.strip().split("\n")[1:]:
        if not ln.strip():
            continue
        parts = [x.strip().strip('"') for x in ln.split(";")]
        if len(parts) < 6:
            continue
        exp, fr, to, typ, vid_id = parts[1], parts[2], parts[3], parts[4], parts[5]
        if labeler != "ALL" and exp != labeler:
            continue
        try:
            fr, to = float(fr), float(to)
        except ValueError:
            continue
        if typ in offtype_to_idx:
            per_video.setdefault(vid_id, []).append((fr, to, offtype_to_idx[typ]))
    if not per_video:
        print(f"  OFT: 无有效 bout (labeler={labeler})")
        return videos
    print(f"  OFT: 使用标注者 '{labeler}'")
    # 每个视频构建
    for vid_id, bb in sorted(per_video.items()):
        video_path = os.path.join(root, f"{vid_id}.mp4")
        if not os.path.exists(video_path):
            print(f"  OFT: 跳过 {vid_id}, 无视频")
            continue
        cap = _cv2.VideoCapture(video_path)
        n = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        labels = _np.full(n, bg, dtype=_np.int64)
        for fr, to, idx in bb:
            if idx == bg:
                continue
            s = max(0, int(fr * fps)); e = min(n - 1, int(to * fps))
            labels[s:e + 1] = idx
        videos.append({"video": video_path, "mouse": vid_id, "source": "oft",
                       "n_frames": n, "labels": labels.tolist()})
        dist = Counter(labels.tolist())
        print(f"  OFT: {vid_id} {n} 帧, 类别分布="
              f"{dict((class_names[i], dist.get(i, 0)) for i in range(len(class_names)))}")
    return videos


def cosine_with_warmup(epoch, epochs, warmup, lr_base):
    if epoch < warmup:
        return lr_base * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, epochs - warmup)
    return lr_base * 0.5 * (1 + np.cos(np.pi * t))


@torch.no_grad()
def evaluate(model, loader, cfg, device, criterion=None, class_names=None):
    """多标签评估。返回 (probs, labels, loss_dict, per_class)。

    对齐 DeepEthogram 官方: 输出每类独立 sigmoid 概率 (非 argmax),
    用逐类阈值二值化 (background=valid_bg), 逐类算 acc/f1/recall/precision。
    带 tqdm 进度条 (val/test 在控制台与日志均可见)。
    """
    model.eval()
    probs, labels = [], []
    loss_acc = None
    tag = getattr(loader, "_eval_tag", "eval")
    n_samples = len(loader.dataset) if hasattr(loader, "dataset") else "?"
    pbar = tqdm(loader, desc=f"[{tag}] evaluating ({n_samples} samples)", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            logits, aux_logits = out
        else:
            logits, aux_logits = out, None
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.cpu().numpy())
        if criterion is not None:
            oh = nn.functional.one_hot(y, num_classes=logits.shape[1]).float()
            loss, ld = criterion(logits.float(), oh, aux_logits=aux_logits, model=model)
            if loss_acc is None:
                loss_acc = {k: 0.0 for k in ld}
                loss_acc["_n"] = 0
            for k, v in ld.items():
                loss_acc[k] += float(v) * x.size(0)
            loss_acc["_n"] += x.size(0)
    P = np.concatenate(probs)
    Y = np.concatenate(labels)
    if loss_acc is not None:
        n = max(1, loss_acc.pop("_n"))
        loss_dict = {k: v / n for k, v in loss_acc.items()}
    else:
        loss_dict = None

    per_class = None
    if class_names is not None:
        thresholds, _ = evaluate_thresholds(Y, P, class_names, bg_idx=len(class_names) - 1)
        pred = threshold_predict(P, thresholds, class_names)
        pred = compute_background(pred, bg_idx=len(class_names) - 1)
        eye = (Y[:, None] == np.arange(len(class_names))).astype(int)
        per_class = binary_class_metrics(eye, pred, class_names)
    return P, Y, loss_dict, per_class


def save_metrics(path, epoch, class_names, per_class, val_f1m, grad_epoch=None,
                 tr_loss=None, val_loss=None,
                 tr_loss_components=None, val_loss_components=None,
                 tr_per_class=None, tr_f1m=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grad_epoch = grad_epoch or {"grad_l2": 0.0, "grad_max": 0.0, "grad_nan": 0}
    tr_per_class = tr_per_class or per_class
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "train_loss", "val_loss", "train_f1_macro", "val_f1_macro",
                        "grad_l2", "grad_max", "grad_nan"]
                       + [f"train_{c}_acc" for c in class_names]
                       + [f"train_{c}_p" for c in class_names]
                       + [f"train_{c}_r" for c in class_names]
                       + [f"train_{c}_f1" for c in class_names]
                       + [f"val_{c}_acc" for c in class_names]
                       + [f"val_{c}_p" for c in class_names]
                       + [f"val_{c}_r" for c in class_names]
                       + [f"val_{c}_f1" for c in class_names])
        row = [epoch, f"{tr_loss:.4f}" if tr_loss is not None else "",
               f"{val_loss:.4f}" if val_loss is not None else "",
               f"{tr_f1m:.2f}" if tr_f1m is not None else "",
               f"{val_f1m:.2f}",
               f"{grad_epoch['grad_l2']:.2e}", f"{grad_epoch['grad_max']:.2e}",
               grad_epoch['grad_nan']]
        for src in (tr_per_class, per_class):
            for c in class_names:
                row += [f"{src[c]['accuracy']:.1f}"]
            for c in class_names:
                row += [f"{src[c]['precision']:.1f}"]
            for c in class_names:
                row += [f"{src[c]['recall']:.1f}"]
            for c in class_names:
                row += [f"{src[c]['f1']:.1f}"]
        w.writerow(row)


def make_optimizer(model, base_lr, head_mult, weight_decay):
    head_groups, base_groups = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # head/cpam/stsa 为新增/头部模块, 用高 lr (与 Lightning 分层一致)
        (head_groups if any(k in name for k in ("head", "cpam", "stsa"))
         else base_groups).append(p)
    groups = []
    if base_groups:
        groups.append({"params": base_groups, "lr": base_lr,
                       "weight_decay": weight_decay})
    if head_groups:
        groups.append({"params": head_groups, "lr": base_lr * head_mult,
                       "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def train_stage(model, tr_ld, va_ld, cfg, criterion, class_names, device,
                num_classes, epochs, base_lr, head_mult, warmup, patience,
                ckpt_dir, metrics_csv, tag):
    """跑一个训练阶段 (LP 或 LP-FT)。返回该阶段最优 val F1_macro。"""
    opt = make_optimizer(model, base_lr, head_mult,
                         float(cfg["TRAIN"].get("weight_decay", 0.02)))
    best_f1, best_epoch, bad = 0.0, -1, 0
    for epoch in range(epochs):
        model.train()
        lr_cur = cosine_with_warmup(epoch, epochs, warmup, base_lr)
        # LP 阶段: head 是唯一可训 (gi=0), 不乘 head_mult; LP-FT: head 组乘 head_mult
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_cur * (head_mult if gi >= 1 else 1.0)

        # ---- 训练 (tqdm 实时显示 loss/acc/f1/grad) ----
        tr_loss, n_tr = 0.0, 0
        tr_loss_components = {k: 0.0 for k in
                              ("L_CE", "L_CPAM", "combined", "L2-SP", "total")}
        grad_l2_sum = 0.0; grad_max_sum = 0.0; grad_nan_sum = 0; n_steps = 0
        tr_tp = np.zeros(num_classes); tr_fp = np.zeros(num_classes)
        tr_fn = np.zeros(num_classes); tr_n = 0
        pbar = tqdm(tr_ld, desc=f"[{tag}] ep{epoch+1}/{epochs} lr={lr_cur:.1e}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                logits, aux_logits = out
            else:
                logits, aux_logits = out, None
            oh = nn.functional.one_hot(y, num_classes=num_classes).float()
            # AuxFocalLoss 内部: 主头 Focal + CPAM 条件自适应 (式8-10) + L2-SP
            loss, loss_dict = criterion(logits.float(), oh, aux_logits=aux_logits, model=model)
            data_loss = loss_dict.get("combined", loss).detach()
            opt.zero_grad()
            loss.backward()
            # ---- 梯度健康度详细记录 ----
            g_l2 = 0.0; g_max = 0.0; g_nan = 0; g_cnt = 0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.float()
                g_l2 += (g ** 2).sum().item()
                g_max = max(g_max, g.abs().max().item())
                g_nan += int(torch.isnan(g).sum().item()) + int(torch.isinf(g).sum().item())
                g_cnt += g.numel()
            grad_l2 = math.sqrt(g_l2)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["TRAIN"]["grad_clip"]))
            opt.step()
            # 实时累计 (train 也用多标签: sigmoid 阈值, 与 val 一致)
            tr_loss += data_loss.item() * x.size(0)
            n_tr += x.size(0)
            for k in tr_loss_components:
                if k in loss_dict:
                    tr_loss_components[k] += float(loss_dict[k]) * x.size(0)
            grad_l2_sum += grad_l2; grad_max_sum += g_max; grad_nan_sum += g_nan
            # train 多标签指标: sigmoid probs 按每类阈值 0.5 二值化 (与 val 同口径)
            tr_probs = torch.sigmoid(logits.detach()).cpu().numpy()
            tr_pred = (tr_probs >= 0.5).astype(int)
            y_np = y.cpu().numpy()
            eye = (y_np[:, None] == np.arange(num_classes)).astype(int)
            for i in range(num_classes):
                t = eye[:, i]; p_ = tr_pred[:, i]
                tr_tp[i] += int(((t == 1) & (p_ == 1)).sum())
                tr_fp[i] += int(((t == 0) & (p_ == 1)).sum())
                tr_fn[i] += int(((t == 1) & (p_ == 0)).sum())
            tr_n += x.size(0)
            # tqdm 瀹炴椂鏄剧ず
            tr_acc = tr_tp.sum() / max(1, tr_n)
            tr_f1s = np.where(tr_tp + tr_fn > 0,
                              2 * tr_tp / (2 * tr_tp + tr_fp + tr_fn + 1e-9), 0.0)
            pbar.set_postfix({"loss": f"{tr_loss/max(1,n_tr):.3f}",
                              "acc": f"{tr_acc*100:.1f}",
                              "f1": f"{tr_f1s.mean()*100:.1f}",
                              "gL2": f"{grad_l2:.1e}"})
            n_steps += 1
        tr_loss /= max(n_tr, 1)
        for k in tr_loss_components:
            tr_loss_components[k] /= max(n_tr, 1)
        tr_acc = tr_tp.sum() / max(1, tr_n)
        tr_per_class = {
            c: {"accuracy": tr_tp[i] / max(1, tr_n) * 100,
                "precision": tr_tp[i] / max(1e-9, tr_tp[i] + tr_fp[i]) * 100,
                "recall": tr_tp[i] / max(1e-9, tr_tp[i] + tr_fn[i]) * 100,
                "f1": 2 * tr_tp[i] / max(1e-9, 2 * tr_tp[i] + tr_fp[i] + tr_fn[i]) * 100}
            for i, c in enumerate(class_names)}
        tr_f1m = f1_macro_from(tr_per_class, class_names)
        tr_f1s = np.where(tr_tp + tr_fn > 0, 2 * tr_tp / (2 * tr_tp + tr_fp + tr_fn + 1e-9), 0.0)
        # 每 epoch 梯度汇总
        grad_epoch = {"grad_l2": grad_l2_sum / max(1, n_steps),
                      "grad_max": grad_max_sum / max(1, n_steps),
                      "grad_nan": int(grad_nan_sum)}

        # val 评估 (DeepEthogram: 每类阈值网格 + valid_bg + mean F1 nobg; 可加时序平滑)
        print(f"[{tag}] ep{epoch+1}/{epochs} 训练完成, 验证中... (val样本={len(va_ld.dataset)})", flush=True)
        va_ld._eval_tag = f"{tag}/val"
        P, Y, val_loss_dict, per_class = evaluate(
            model, va_ld, cfg, device, criterion=criterion, class_names=class_names)
        if cfg["POSTPROCESS"].get("temporal_smooth", False):
            P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
            thresholds, _ = evaluate_thresholds(Y, P, class_names, bg_idx=num_classes - 1)
            pred = threshold_predict(P, thresholds, class_names)
            pred = compute_background(pred, bg_idx=num_classes - 1)
            eye = (Y[:, None] == np.arange(num_classes)).astype(int)
            per_class = binary_class_metrics(eye, pred, class_names)
        val_f1m = f1_macro_from(per_class, class_names)
        # DeepEthogram 官方选模型指标: 去 background 的类平均 F1 (mean F1 nobg)
        val_f1m_nobg = float(np.mean(
            [per_class[c]["f1"] for c in class_names[:-1]])) if len(class_names) > 1 else val_f1m
        # 阈值无关的 AUROC (辅助参考)
        val_auroc = macro_auroc(P, Y, class_names)
        # select_metric: f1_nobg(官方) / auroc / auprc / f1
        sel = cfg["TRAIN"].get("select_metric", "f1_nobg")
        if sel == "auprc":
            sel_metric = macro_auprc(P, Y, class_names)
        elif sel == "f1":
            sel_metric = val_f1m
        elif sel == "auroc":
            sel_metric = val_auroc
        else:  # f1_nobg (官方 DeepEthogram)
            sel_metric = val_f1m_nobg
        save_metrics(metrics_csv, epoch, class_names, per_class, val_f1m, grad_epoch,
                     tr_loss=tr_loss, val_loss=val_loss_dict.get("total", 0.0) if val_loss_dict else 0.0,
                     tr_loss_components=tr_loss_components,
                     val_loss_components=val_loss_dict,
                     tr_per_class=tr_per_class, tr_f1m=tr_f1m)

        # ---- 每 epoch 结束后逐类 acc/f1/recall/precision + 梯度汇总 ----
        print(f"\n[{tag}] ep{epoch+1}/{epochs} | TRAIN loss={tr_loss:.3f} f1_macro={tr_f1m:.2f} "
              f"| VAL loss={(val_loss_dict.get('total',0.0) if val_loss_dict else 0.0):.3f} "
              f"f1_macro={val_f1m:.2f} (nobg={val_f1m_nobg:.2f}) "
              f"AUROC={val_auroc:.3f} | "
              f"grad_L2={grad_epoch['grad_l2']:.1e} grad_max={grad_epoch['grad_max']:.1e} "
              f"grad_NaN={grad_epoch['grad_nan']}", flush=True)
        print(f"[{tag}] TRAIN {'Class':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}", flush=True)
        for c in class_names:
            pc = tr_per_class[c]
            print(f"[{tag}]       {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%", flush=True)
        print(f"[{tag}] VAL   {'Class':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}", flush=True)
        for c in class_names:
            pc = per_class[c]
            print(f"[{tag}]       {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%", flush=True)

        torch.save({"epoch": epoch, "stage": tag, "model_state": model.state_dict()},
                   os.path.join(ckpt_dir, f"{tag}_epoch_{epoch:03d}.pt"))
        if sel_metric > best_f1:
            best_f1, best_epoch, bad = sel_metric, epoch, 0
            torch.save({"epoch": epoch, "stage": tag, "model_state": model.state_dict(),
                        "best_metric": best_f1, "val_f1m": val_f1m,
                        "val_f1m_nobg": val_f1m_nobg},
                       os.path.join(ckpt_dir, f"{tag}_best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] 鏃╁仠 (best_{sel}={best_f1:.3f} ep{best_epoch+1})")
                break
    print(f"[{tag}] 瀹屾垚, best {sel}={best_f1:.3f} ep{best_epoch+1}")
    return best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lp-epochs", type=int, default=None)
    ap.add_argument("--subset-videos", type=int, default=None)
    ap.add_argument("--log-file", default=None,
                    help="将控制台输出同时写入该文件 (默认 outputs/logs/train_<时间>.log)")
    # 消融开关: 显式覆盖 config 中的 stsa/cpam; 未给出则用 config 值
    ap.add_argument("--stsa", type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                    default=None, help="override STSA on/off (None=use config)")
    ap.add_argument("--cpam", type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                    default=None, help="override CPAM on/off (None=use config)")
    # 输出 tag: 每个消融写独立 checkpoint/metric/log 目录
    ap.add_argument("--tag", default=None,
                    help="输出目录后缀 (outputs/checkpoints_<tag> 等); 默认 config 目录")
    ap.add_argument("--oft-epochs", type=int, default=None,
                    help="OFT 预训练 epoch 数 (覆盖 config)")
    args = ap.parse_args()

    # 自动落盘日志 (PyCharm 运行时也能生成, 便于后期优化)
    import time as _t, io as _io
    log_path = args.log_file or os.path.join(
        ROOT, "outputs", "logs", f"train_{_t.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _logf = open(log_path, "a", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    class _Tee:
        """同时写控制台 + 日志文件。不依赖 .buffer, 兼容 IDE 运行控制台。"""
        def __init__(self, orig):
            self._orig = orig
        def write(self, s):
            try:
                _logf.write(s); _logf.flush()
            except Exception:
                pass
            try:
                return self._orig.write(s)
            except Exception:
                return len(s)
        def flush(self):
            try: self._orig.flush()
            except Exception: pass
        def isatty(self):
            try: return self._orig.isatty()
            except Exception: return False
    sys.stdout = _Tee(_orig_stdout)
    sys.stderr = _Tee(_orig_stderr)
    print(f"[log] 控制台输出已写入: {log_path}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # 消融开关覆盖
    if args.stsa is not None:
        cfg["MODEL"]["stsa"] = args.stsa
    if args.cpam is not None:
        cfg["MODEL"]["cpam"] = args.cpam
    if args.oft_epochs is not None:
        cfg.setdefault("TRAIN", {}).setdefault("oft", {})["epochs"] = args.oft_epochs

    # 输出 tag: 独立目录, 避免消融间互相覆盖
    if args.tag:
        for k in ("checkpoint_dir", "metric_dir", "log_dir"):
            if k in cfg["OUTPUT"]:
                cfg["OUTPUT"][k] = f"{cfg['OUTPUT'][k]}_{args.tag}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    frame_window = int(cfg["CLIP"]["num_input_frames"])

    # 数据划分 (按小鼠, 已由 02 脚本生成)
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    train_v = split["train"]
    val_v = split["val"]
    if args.subset_videos:
        train_v = train_v[:args.subset_videos]
        val_v = val_v[:max(1, args.subset_videos // 4)]
    print(f"train {len(train_v)} 视频 / val {len(val_v)} 视频")

    # pos_weight (帧级类别不平衡)
    train_ds = FrameLevelDataset(train_v, cfg, split="train",
                                 class_names=class_names, frame_window=frame_window)
    val_ds = FrameLevelDataset(val_v, cfg, split="val",
                               class_names=class_names, frame_window=frame_window)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    print("类别分布:", dict(zip(class_names, [int(x) for x in cc])))
    print("pos_weight:", [round(x, 2) for x in pw.tolist()])

    # 模型
    model = build_model(cfg, num_classes).to(device)
    print(f"[ABLATION] stsa={bool(cfg['MODEL']['stsa'])} cpam={bool(cfg['MODEL']['cpam'])} "
          f"tag={args.tag or '(default)'}")
    model.set_head(num_classes, class_counts=cc)

    bs = int(cfg["TRAIN"]["batch_size"])
    nw = int(cfg["TRAIN"]["num_workers"])
    use_balanced = bool(cfg["TRAIN"].get("sampler_balanced", True))
    tr_ld = build_loader(train_ds, cfg, split="train", batch_size=bs,
                         num_workers=nw, balanced=use_balanced,
                         seed=int(cfg["TRAIN"].get("seed", 42)))
    if use_balanced:
        print(f"[sampler] 训练启用 Swin-Panda 均匀过采样 (稀有类补到平均值), "
              f"平衡窗口总数={len(train_ds.fixed_windows)}")
    va_ld = build_loader(val_ds, cfg, split="val", batch_size=bs,
                         num_workers=nw, shuffle=False)

    warmup = int(cfg["TRAIN"].get("warmup_epochs", 8))
    patience = int(cfg["TRAIN"].get("early_stop_patience", 12))
    metric_dir = cfg["OUTPUT"]["metric_dir"]
    ckpt_dir = cfg["OUTPUT"]["checkpoint_dir"]
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = os.path.join(metric_dir, "train_metrics.csv")
    weight_decay = float(cfg["TRAIN"].get("weight_decay", 0.02))

    def _make_criterion(n_cls, pw_vec):
        return AuxFocalLoss(
            num_classes=n_cls,
            pos_weight=pw_vec.to(device),
            gamma=float(cfg["LOSS"].get("binary_gamma", 0.5)),
            alpha=float(cfg["MODEL"].get("cpam_alpha", 0.9)),
            beta=float(cfg["MODEL"].get("cpam_beta", 0.1)),
            l2sp_weight=float(cfg["TRAIN"].get("l2sp_weight", 0.0)),
            label_smoothing=float(cfg["LOSS"].get("binary_label_smoothing", 0.05)),
        )

    # ================= OFT 预训练阶段 (Grooming 粗分类) =================
    # 严格对齐 SwinPanda 冻结协议: 冻结前3stage, 训 Stage4+STSA+CPAM+head
    oft = cfg["TRAIN"].get("oft", {})
    if bool(oft.get("enable", False)):
        oft_classes = oft.get("classes", ["background", "Supported", "Unsupported", "Grooming", "Jumping"])
        oft_root = cfg["DATA"].get("offtrain_root", "E:/tsy/downloaddatasets2")
        oft_v = build_oft_videos(oft_root, class_names=oft_classes)
        if oft_v:
            print(f"[OFT] Grooming粗分类预训练: {len(oft_v)} 视频, 类别={oft_classes}")
            model.unfreeze_all()
            model.set_num_classes(len(oft_classes))
            oft_pw, oft_cc = make_pos_weight_from_videos(oft_v, oft_classes)
            oft_ds = FrameLevelDataset(oft_v, cfg, split="train",
                                       class_names=oft_classes, frame_window=frame_window)
            oft_ld = build_loader(oft_ds, cfg, split="train", batch_size=bs,
                                  num_workers=nw, balanced=use_balanced,
                                  seed=int(cfg["TRAIN"].get("seed", 42)))
            oft_crit = _make_criterion(len(oft_classes), oft_pw)
            # 严格 SwinPanda: 冻结前3stage, 训 Stage4+STSA+CPAM+head
            model.freeze_first_stages(3)
            n_st4 = sum(p.numel() for p in model.stages[3].parameters())
            n_stsa = (sum(p.numel() for p in model.stsa3.parameters()) if model.stsa3 else 0) + \
                     (sum(p.numel() for p in model.stsa4.parameters()) if model.stsa4 else 0)
            n_cpam = sum(p.numel() for p in model.cpam.parameters()) if model.cpam else 0
            n_head = sum(p.numel() for p in model.head.parameters())
            print(f"[OFT] 冻结前3stage, 训 Stage4({n_st4})+STSA({n_stsa})+CPAM({n_cpam})+head({n_head})")
            # OFT 验证用"小分层验证集", 避免对全量 oft_ld 再跑一遍 (验证静默且耗时)
            oft_val_ds = FrameLevelDataset(oft_v, cfg, split="val",
                                           class_names=oft_classes, frame_window=frame_window)
            oft_val_ld = build_loader(oft_val_ds, cfg, split="val", batch_size=bs,
                                      num_workers=nw, shuffle=False)
            print(f"[OFT] 验证集: {len(oft_val_ds)} 样本 (小, 快速)")
            oft_best = train_stage(
                model, oft_ld, oft_val_ld, cfg, oft_crit, oft_classes, device,
                len(oft_classes),
                epochs=int(oft.get("epochs", 10)),
                base_lr=float(oft.get("lr", 2e-4)),
                head_mult=float(oft.get("head_lr_mult", 10.0)),
                warmup=min(warmup, int(oft.get("epochs", 10))), patience=patience,
                ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="oft")
            torch.save({"epoch": "oft", "stage": "oft", "model_state": model.state_dict()},
                       os.path.join(ckpt_dir, "oft_pretrained.pt"))
            print(f"[OFT] 预训练完成, 保存 oft_pretrained.pt. 切换到正式 5 类训练..")
            # 切回正式 5 类 head + CPAM aux_head
            model.set_num_classes(num_classes, class_counts=cc)
            model.unfreeze_all()

    # 正式 5 类损失
    criterion = _make_criterion(num_classes, pw)

    # ---- 训练策略: VideoGLUE/LP-FT 两段式 (linear probe -> LP-FT) ----
    lp = cfg["TRAIN"].get("lp", {})
    lpft = cfg["TRAIN"].get("lpft", {})
    do_lp = bool(lp.get("enable", False))
    if do_lp:
        # 阶段1: Linear Probe -- 冻结主干(含STSA), 只训 head + cpam (新增模块保持可训)
        model.freeze_backbone()
        n_head_params = sum(p.numel() for p in model.head.parameters())
        n_cpam_params = (sum(p.numel() for p in model.cpam.parameters())
                         if model.cpam is not None else 0)
        n_stsa_params = (sum(p.numel() for p in model.stsa3.parameters())
                         if model.stsa3 is not None else 0) + \
                        (sum(p.numel() for p in model.stsa4.parameters())
                         if model.stsa4 is not None else 0)
        print(f"[LP] 冻结主干, 只训 head ({n_head_params}) + "
              f"stsa ({n_stsa_params}) + cpam ({n_cpam_params}) 参数")
        train_stage(
            model, tr_ld, va_ld, cfg, criterion, class_names, device,
            num_classes,
            epochs=args.lp_epochs or int(lp.get("epochs", 5)),
            base_lr=float(lp.get("lr", 5e-4)), head_mult=1.0,
            warmup=min(warmup, int(lp.get("epochs", 5))), patience=patience,
            ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lp")

    # 阶段2: LP-FT -- 冻结前 3 个 stage, 解冻 Stage4 + STSA + CPAM + head (论文: 训 Stage4+新增模块)
    ft_epochs = args.epochs or int(lpft.get("epochs", int(cfg["TRAIN"]["epochs"])))
    if do_lp:
        model.freeze_first_stages(3)   # 冻结 Stage1-3, 保持 Stage4+STSA+CPAM+head 可训
    best_f1 = train_stage(
        model, tr_ld, va_ld, cfg, criterion, class_names, device,
        num_classes,
        epochs=ft_epochs,
        base_lr=float(lpft.get("lr", float(cfg["TRAIN"]["lr"]))),
        head_mult=float(lpft.get("head_lr_mult", cfg["TRAIN"].get("head_lr_mult", 10.0))),
        warmup=warmup, patience=patience,
        ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lpft")

    print(f"\n瀹屾垚, best val f1_macro={best_f1:.2f}")
    print(f"鎸囨爣 -> {metrics_csv}")

    # ---- 训练结束后, 用最佳 LP-FT 权重在独立 test 集评估 ----
    best_ckpt = os.path.join(ckpt_dir, "lpft_best.pt")
    if os.path.exists(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=False)
        print(f"  加载 {best_ckpt} (epoch {ck.get('epoch','?')})")

    def _eval_set(videos, name, out_csv):
        if not videos:
            print(f"\n[{name}] 无视频, 跳过")
            return
        print(f"\n[{name}] 评估最佳模型 on {len(videos)} 视频..")
        ds = FrameLevelDataset(videos, cfg, split="val",
                               class_names=class_names, frame_window=frame_window)
        ld = build_loader(ds, cfg, split="val", batch_size=bs,
                          num_workers=nw, shuffle=False)
        ld._eval_tag = name
        P, Y, loss_dict, tpc = evaluate(model, ld, cfg, device,
                                        criterion=criterion, class_names=class_names)
        if cfg["POSTPROCESS"].get("temporal_smooth", False):
            P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
            thresholds, _ = evaluate_thresholds(Y, P, class_names, bg_idx=num_classes - 1)
            pred = threshold_predict(P, thresholds, class_names)
            pred = compute_background(pred, bg_idx=num_classes - 1)
            eye = (Y[:, None] == np.arange(num_classes)).astype(int)
            tpc = binary_class_metrics(eye, pred, class_names)
        tf1 = f1_macro_from(tpc, class_names)
        tl = loss_dict.get("total", 0.0) if loss_dict else 0.0
        t_auroc = macro_auroc(P, Y, class_names)
        print(f"[{name}] TEST loss={tl:.4f} F1_macro={tf1:.2f} AUROC={t_auroc:.3f}")
        print(f"[{name}] {'Class':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}")
        for c in class_names:
            pc = tpc[c]
            print(f"[{name}] {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%")
        save_metrics(out_csv, 0, class_names, tpc, tf1,
                     tr_loss=tl, val_loss=tl, tr_per_class=tpc, tr_f1m=tf1)

    # 内置 test 集 (split.json)
    test_v = split.get("test") or []
    _eval_set(test_v, "TEST", os.path.join(metric_dir, "test_metrics.csv"))

    # 外部泛化测试集 datasetsopto (Defecate/Jump 不纳入: 映射为 background)
    opto_root = cfg["DATA"].get("opto_test_root", "E:/tsy/datasetsopto")
    if os.path.isdir(opto_root):
        opto_v = build_videos_from_root(opto_root, "opto")
        _eval_set(opto_v, "TEST_OPTO", os.path.join(metric_dir, "test_opto_metrics.csv"))
    else:
        print(f"\n[TEST_OPTO] 未找到外部测试目录: {opto_root}")
    print(f"\n完成, best val f1_macro={best_f1:.2f}")
    print(f"指标 -> {metrics_csv}")


if __name__ == "__main__":
    main()
