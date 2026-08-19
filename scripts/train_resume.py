# -*- coding: utf-8 -*-
"""train_resume.py - 完整训练管线 + 断电续训 (auto-resume).

完整流程: OFT 预训练 -> LP(冻前3stage训Stage4+head+STSA+CPAM) -> LP-FT(正式5类微调).
本文件自包含全部训练/评估函数 (原 scripts/train.py 已合并至此)。

自动跳过已完成阶段:
  1. OFT 阶段:  若 oft_pretrained.pt 存在 -> 跳过, 否则训练.
  2. LP 阶段:   若 lp_best.pt 存在 -> 跳过, 否则训练.
  3. LP-FT 阶段: 取最新 lpft_epoch_XXX.pt / lpft_best.pt 加载继续; 否则从头训.

用法:
  python scripts/train_resume.py                     # 自动续训
  python scripts/train_resume.py --tag smoke         # 续到独立 smoke 目录
  python scripts/train_resume.py --from-ckpt PATH --epochs 50
"""
import os, sys, json, glob, csv, argparse, math
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

from data.dataset import FrameLevelDataset, make_pos_weight_from_videos
from data.dataloaders import build_loader
from models.tadaconv_model import TAdaConvModel, load_pretrained
from losses.auxiliary import AuxFocalLoss
from postprocess.thresholds import (evaluate_thresholds, threshold_predict,
                                    compute_background, min_bout_per_behavior,
                                    binary_metrics, smooth_probabilities,
                                    hmm_smooth, bout_stats)
from postprocess.ethogram_metrics import ethogram_metrics  # DeepEthogram 官方指标迁移
from torch.utils.data import DataLoader  # noqa: F401


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
        use_checkpoint=bool(m.get("use_checkpoint", False)),
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
def evaluate(model, loader, cfg, device, criterion=None, class_names=None, thresholds=None):
    """多标签评估。返回 (probs, labels, loss_dict, per_class, used_thresholds)。

    对齐 DeepEthogram 官方: 输出每类独立 sigmoid 概率 (非 argmax),
    用逐类阈值二值化 (background=valid_bg), 逐类算 acc/f1/recall/precision。
    带 tqdm 进度条 (val/test 在控制台与日志均可见)。

    Args:
        thresholds: 外部传入的固定阈值 {class: float}。为 None 时在本数据集上
                    网格搜索最优阈值 (仅用于参考上限, 不应用于模型选择)。
    """
    model.eval()
    probs, labels = [], []
    loss_acc = None
    tag = getattr(loader, "_eval_tag", "eval")
    # background 类索引 (不硬编码为末类: OFT 的 background=0, openfield=末类)
    bg_idx = (class_names.index("background") if class_names and "background" in class_names
              else (len(class_names) - 1 if class_names else None))
    n_samples = len(loader.dataset) if hasattr(loader, "dataset") else "?"
    total_batches = max(1, len(loader))
    pbar = tqdm(loader, desc=f"[{tag}] evaluating ({n_samples} samples)", leave=True)
    seen = 0
    for bi, (x, y) in enumerate(pbar):
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
            run_loss = loss_acc.get("total", 0.0) / max(1, loss_acc["_n"])
            pbar.set_postfix({"loss": f"{run_loss:.4f}"}, refresh=True)
        # 每 10% 打一条详细日志 (控制台+log), 便于核对每批进度
        seen += x.size(0)
        if (bi + 1) % max(1, total_batches // 10) == 0 or bi == total_batches - 1:
            rl = (loss_acc.get("total", 0.0) / max(1, loss_acc["_n"])) if loss_acc else 0.0
            print(f"  [{tag}] 已处理 {seen}/{n_samples} 样本 (batch {bi+1}/{total_batches}) "
                  f"avg_loss={rl:.4f}", flush=True)
    pbar.close()
    P = np.concatenate(probs)
    Y = np.concatenate(labels)
    if loss_acc is not None:
        n = max(1, loss_acc.pop("_n"))
        loss_dict = {k: v / n for k, v in loss_acc.items()}
    else:
        loss_dict = None
    print(f"  [{tag}] 评估完成: {len(Y)} 样本, loss={loss_dict.get('total', float('nan')) if loss_dict else float('nan'):.4f}", flush=True)

    per_class = None
    used_thresholds = thresholds
    epoch_metrics = None
    if class_names is not None:
        # DeepEthogram 官方 evaluate_thresholds: 每类 argmax-F1 调阈值 + valid_bg(background=NOT others)
        # 在给定数据上直接调优并报指标 (官方行为), 阈值随 split 独立。
        epoch_metrics, per_class, used_thresholds = ethogram_metrics(P, Y, class_names, bg_idx=bg_idx)
        cnt = np.bincount(Y.astype(np.int64), minlength=len(class_names))
        thr_str = ", ".join(f"{c}={float(used_thresholds[i]):.3f}" for i, c in enumerate(class_names))
        cnt_str = ", ".join(f"{c}={int(cnt[i])}" for i, c in enumerate(class_names))
        print(f"  [{tag}] 每类样本数: {cnt_str}", flush=True)
        print(f"  [{tag}] 每类官方阈值(argmax-F1+valid_bg): {thr_str}", flush=True)
    return P, Y, loss_dict, per_class, used_thresholds, epoch_metrics


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
        tr_fn = np.zeros(num_classes); tr_tn = np.zeros(num_classes); tr_n = 0
        tr_corr = 0  # argmax 预测 == 真实标签 的样本数 (诚实总准确率)
        # background 类索引 (不硬编码末类: OFT 的 background=0)
        bg_idx = (class_names.index("background") if class_names and "background" in class_names
                  else (num_classes - 1 if class_names else None))
        # 累计训练集概率 (epoch 末用于调阈值, 避免验证集自调过拟合)
        tr_all_probs = []; tr_all_labels = []
        pbar = tqdm(tr_ld, desc=f"[{tag}] ep{epoch+1}/{epochs} lr={lr_cur:.1e}", leave=False)
        n_batches_total = max(1, len(tr_ld))
        for bi, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            grad_l2 = 0.0   # postfix 用 (非累积步显示 0)
            out = model(x)
            if isinstance(out, tuple):
                logits, aux_logits = out
            else:
                logits, aux_logits = out, None
            oh = nn.functional.one_hot(y, num_classes=num_classes).float()
            # AuxFocalLoss 内部: 主头 Focal + CPAM 条件自适应 (式8-10) + L2-SP
            loss, loss_dict = criterion(logits.float(), oh, aux_logits=aux_logits, model=model)
            data_loss = loss_dict.get("combined", loss).detach()
            # 梯度累积: 每 grad_accum 步累计梯度, 最后一步才更新权重 (有效 batch = bs*accum)
            accum = int(cfg["TRAIN"].get("grad_accum", 1))
            loss = loss / accum
            is_accum_step = ((bi + 1) % accum == 0) or ((bi + 1) == n_batches_total)
            loss.backward()
            if is_accum_step:
                # ---- 梯度健康度详细记录 (每次真正更新时) ----
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
                grad_l2_sum += grad_l2; grad_max_sum += g_max; grad_nan_sum += g_nan
                n_steps += 1
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["TRAIN"]["grad_clip"]))
                opt.step()
                opt.zero_grad()
            # 实时累计 (train 也用多标签: sigmoid 阈值, 与 val 一致)
            tr_loss += data_loss.item() * x.size(0)
            n_tr += x.size(0)
            for k in tr_loss_components:
                if k in loss_dict:
                    tr_loss_components[k] += float(loss_dict[k]) * x.size(0)
            # train 多标签指标: sigmoid probs 按每类阈值 0.5 二值化 (与 val 同口径)
            tr_probs = torch.sigmoid(logits.detach()).cpu().numpy()
            tr_all_probs.append(tr_probs)
            tr_pred = (tr_probs >= 0.5).astype(int)
            y_np = y.cpu().numpy()
            tr_all_labels.append(y_np)
            eye = (y_np[:, None] == np.arange(num_classes)).astype(int)
            for i in range(num_classes):
                t = eye[:, i]; p_ = tr_pred[:, i]
                tr_tp[i] += int(((t == 1) & (p_ == 1)).sum())
                tr_fp[i] += int(((t == 0) & (p_ == 1)).sum())
                tr_fn[i] += int(((t == 1) & (p_ == 0)).sum())
                tr_tn[i] += int(((t == 0) & (p_ == 0)).sum())
            tr_n += x.size(0)
            tr_corr += int((tr_probs.argmax(axis=1) == y_np).sum())
            # tqdm 实时显示 (总准确率 = argmax, 不被真阴性注水)
            tr_acc = tr_corr / max(1, tr_n)
            tr_f1s = np.where(tr_tp + tr_fn > 0,
                              2 * tr_tp / (2 * tr_tp + tr_fp + tr_fn + 1e-9), 0.0)
            pbar.set_postfix({"loss": f"{tr_loss/max(1,n_tr):.3f}",
                              "acc": f"{tr_acc*100:.1f}",
                              "f1": f"{tr_f1s.mean()*100:.1f}",
                              "gL2": f"{grad_l2:.1e}"})
        tr_loss /= max(n_tr, 1)
        for k in tr_loss_components:
            tr_loss_components[k] /= max(n_tr, 1)
        tr_acc = tr_corr / max(1, tr_n)
        tr_per_class = {
            c: {"accuracy": (tr_tp[i] + tr_tn[i]) / max(1, tr_n) * 100,
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

        # ---- 官方 DeepEthogram: 训练集上逐类 argmax-F1 调阈值 + valid_bg, 报训练集指标 ----
        tr_P = np.concatenate(tr_all_probs)
        tr_Y = np.concatenate(tr_all_labels)
        tr_em, tr_per_class, train_thresholds = ethogram_metrics(tr_P, tr_Y, class_names, bg_idx=bg_idx)
        thr_str = ", ".join(f"{c}={float(train_thresholds[i]):.3f}" for i, c in enumerate(class_names))
        print(f"[{tag}] ep{epoch+1}/{epochs} 训练集官方阈值(argmax-F1+valid_bg): {thr_str}", flush=True)
        tr_f1m = tr_em["f1_class_mean"]

        # val 评估 (官方 evaluate_thresholds: 在 val 上逐类 argmax-F1 调阈值 + valid_bg)
        print(f"[{tag}] ep{epoch+1}/{epochs} 训练完成, 验证中... (val样本={len(va_ld.dataset)})", flush=True)
        va_ld._eval_tag = f"{tag}/val"
        P, Y, val_loss_dict, per_class, val_thresholds, val_em = evaluate(
            model, va_ld, cfg, device, criterion=criterion, class_names=class_names)
        # DeepEthogram 官方上报指标 (eLife Evaluation 段):
        #   accuracy_overall=所有类混淆矩阵求和后的二元acc; f1_class_mean=全类(含bg) macro-F1;
        #   auroc_class_mean=逐类AUROC均值; mAP_class_mean=逐类mAP均值
        val_f1m = val_em["f1_class_mean"]
        val_f1m_nobg = val_em["f1_class_mean_nobg"]     # 按 bg_idx 排除 background (修 OFT bg=0 的 bug)
        val_auroc = val_em["auroc_class_mean"]
        val_overall_acc = val_em["accuracy_overall"]
        val_mAP = val_em["mAP_class_mean"]
        # select_metric: auroc(默认) / f1_nobg(DeepEthogram HPO目标) / f1 / mAP
        sel = cfg["TRAIN"].get("select_metric", "auroc")
        if sel == "f1":
            sel_metric = val_f1m
        elif sel == "f1_nobg":
            sel_metric = val_f1m_nobg
        elif sel == "mAP":
            sel_metric = val_mAP
        else:  # auroc (默认, 阈值无关不过拟合)
            sel_metric = val_auroc
        save_metrics(metrics_csv, epoch, class_names, per_class, val_f1m, grad_epoch,
                     tr_loss=tr_loss, val_loss=val_loss_dict.get("total", 0.0) if val_loss_dict else 0.0,
                     tr_loss_components=tr_loss_components,
                     val_loss_components=val_loss_dict,
                     tr_per_class=tr_per_class, tr_f1m=tr_f1m)

        # ---- 每 epoch 结束后: DeepEthogram 官方指标 (overall acc + 全类macro-F1 + macro-AUROC) ----
        print(f"\n[{tag}] ep{epoch+1}/{epochs} | TRAIN loss={tr_loss:.3f} F1_macro(调优阈)={tr_f1m:.2f} "
              f"| VAL loss={(val_loss_dict.get('total',0.0) if val_loss_dict else 0.0):.3f} "
              f"overall_acc={val_overall_acc*100:.1f}% "
              f"F1_macro(全类)={val_f1m:.2f} AUROC(macro)={val_auroc:.3f} "
              f"[选择目标]={sel}={sel_metric:.3f} "
              f"| grad_L2={grad_epoch['grad_l2']:.1e} grad_max={grad_epoch['grad_max']:.1e} "
              f"grad_NaN={grad_epoch['grad_nan']}", flush=True)
        print(f"[{tag}] TRAIN (阈值=训练集调优) {'Class':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}", flush=True)
        for c in class_names:
            pc = tr_per_class[c]
            print(f"[{tag}]       {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%", flush=True)
        print(f"[{tag}] VAL   (阈值=训练集调优+平滑) {'Class':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}", flush=True)
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



def make_criterion(cfg, num_classes, pw, device):
    # DeepEthogram 官方: weighted binary focal (γ=1) + pos_weight=(neg/pos)^0.25
    return AuxFocalLoss(
        num_classes=num_classes, pos_weight=pw.to(device),
        gamma=float(cfg["LOSS"].get("binary_gamma", 1.0)),
        alpha=float(cfg["MODEL"].get("cpam_alpha", 0.9)),
        beta=float(cfg["MODEL"].get("cpam_beta", 0.1)),
        l2sp_weight=float(cfg["TRAIN"].get("l2sp_weight", 0.0)))


def load_weights(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model_state"]
    cur = model.state_dict()
    # 只加载"键存在且形状一致"的权重, 跳过 head/aux_head 类别数不同(4<->5)等不匹配键
    # (load_state_dict 即使 strict=False 对形状不匹配仍会报错)
    new = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(new, strict=False)
    n_skip = len(sd) - len(new)
    print(f"[resume] 加载 {os.path.basename(path)} (epoch={ck.get('epoch','?')} "
          f"stage={ck.get('stage','?')}) 兼容 {len(new)}/{len(sd)} 参数, 跳过 {n_skip} (类别数/形状不匹配)",
          flush=True)
    return model


def latest_lpft_ckpt(ckpt_dir):
    """返回最新 lpft_epoch_XXX.pt 或 lpft_best.pt. (XXX 最大)."""
    pats = glob.glob(os.path.join(ckpt_dir, "lpft_epoch_*.pt")) + \
           glob.glob(os.path.join(ckpt_dir, "lpft_best.pt"))
    if not pats:
        return None
    def _key(p):
        b = os.path.basename(p)
        if "best" in b:
            return -1
        try:
            return int(b.split("_epoch_")[1].split(".")[0])
        except Exception:
            return -2
    return max(pats, key=_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--tag", default=None, help="输出目录后缀 (续对应 tag 的 checkpoint)")
    ap.add_argument("--from-ckpt", default=None, help="强制从指定 ckpt 续训 LP-FT")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 LP-FT 剩余 epoch 数")
    ap.add_argument("--log-file", default=None, help="日志文件路径 (默认 outputs/logs/train2_<时间>.log)")
    args = ap.parse_args()

    # ---- 控制台 + 日志文件双写 (tqdm 与 print 都写入文件, 控制台失效也不丢进度) ----
    import time as _t, io as _io
    log_path = args.log_file or os.path.join(
        ROOT, "outputs", "logs", f"train2_{_t.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _logf = open(log_path, "a", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr

    class _Tee:
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
    print(f"[log] 控制台输出已写入: {log_path}", flush=True)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.tag:
        for k in ("checkpoint_dir", "metric_dir", "log_dir"):
            cfg["OUTPUT"][k] = f"{cfg['OUTPUT'][k]}_{args.tag}"
    ckpt_dir = cfg["OUTPUT"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    frame_window = int(cfg["CLIP"]["num_input_frames"])
    bs = int(cfg["TRAIN"]["batch_size"])
    nw = int(cfg["TRAIN"]["num_workers"])
    seed = int(cfg["TRAIN"].get("seed", 42))
    warmup = int(cfg["TRAIN"].get("warmup_epochs", 8))
    patience = int(cfg["TRAIN"].get("early_stop_patience", 12))

    print(f"[resume] device={device} ckpt_dir={ckpt_dir}")

    # 数据 (7:2:1 split 已由 02 重建)
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json"), encoding="utf-8"))
    train_v, val_v = split["train"], split["val"]
    print(f"[resume] train {len(train_v)} / val {len(val_v)} 视频")

    train_ds = FrameLevelDataset(train_v, cfg, split="train", class_names=class_names,
                                 frame_window=frame_window)
    val_ds = FrameLevelDataset(val_v, cfg, split="val", class_names=class_names,
                               frame_window=frame_window)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    use_balanced = bool(cfg["TRAIN"].get("sampler_balanced", True))
    tr_ld = build_loader(train_ds, cfg, split="train", batch_size=bs, num_workers=nw,
                         balanced=use_balanced, seed=seed)
    va_ld = build_loader(val_ds, cfg, split="val", batch_size=bs, num_workers=nw, shuffle=False)
    metrics_csv = os.path.join(cfg["OUTPUT"]["metric_dir"], "train_metrics.csv")

    model = build_model(cfg, num_classes).to(device)
    print(f"[resume] 模型参数: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    oft = cfg["TRAIN"].get("oft", {})
    lp = cfg["TRAIN"].get("lp", {})
    lpft = cfg["TRAIN"].get("lpft", {})

    # ============ OFT 阶段 (跳过若已完成) ============
    oft_done = os.path.exists(os.path.join(ckpt_dir, "oft_pretrained.pt"))
    if bool(oft.get("enable", False)) and not oft_done:
        oft_classes = oft.get("classes", ["background", "Supported", "Unsupported", "Grooming"])
        oft_root = cfg["DATA"].get("offtrain_root", "E:/tsy/downloaddatasets2")
        oft_v = build_oft_videos(oft_root, class_names=oft_classes)
        if oft_v:
            # 按视频留出 OFT 验证集 (消除数据泄漏): 确定性取 20% 视频做 val
            oft_v_sorted = sorted(oft_v, key=lambda v: v["video"])
            n_oft_val = max(1, int(len(oft_v_sorted) * 0.2))
            oft_val_v = oft_v_sorted[:n_oft_val]
            oft_train_v = oft_v_sorted[n_oft_val:]
            print(f"[resume] OFT 预训练: train {len(oft_train_v)} 视频 / val {len(oft_val_v)} 视频 "
                  f"(按视频不重叠)", flush=True)
            model.unfreeze_all()
            model.set_num_classes(len(oft_classes))
            oft_pw, _ = make_pos_weight_from_videos(oft_train_v, oft_classes)
            oft_ds = FrameLevelDataset(oft_train_v, cfg, split="train", class_names=oft_classes,
                                       frame_window=frame_window)
            oft_ld = build_loader(oft_ds, cfg, split="train", batch_size=bs, num_workers=nw,
                                  balanced=use_balanced, seed=seed)
            # OFT 验证用"留出视频的小分层验证集", 避免泄漏且快速
            oft_val_ds = FrameLevelDataset(oft_val_v, cfg, split="val", class_names=oft_classes,
                                           frame_window=frame_window)
            oft_val_ld = build_loader(oft_val_ds, cfg, split="val", batch_size=bs,
                                      num_workers=nw, shuffle=False)
            print(f"[resume] OFT 验证集: {len(oft_val_ds)} 样本 (留出视频, 无泄漏)", flush=True)
            model.freeze_first_stages(3)
            train_stage(model, oft_ld, oft_val_ld, cfg,
                        make_criterion(cfg, len(oft_classes), oft_pw, device),
                        oft_classes, device, len(oft_classes),
                        epochs=int(oft.get("epochs", 5)),
                        base_lr=float(oft.get("lr", 2e-4)),
                        head_mult=float(oft.get("head_lr_mult", 10.0)),
                        warmup=min(warmup, int(oft.get("epochs", 5))), patience=patience,
                        ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="oft")
            torch.save({"epoch": "oft", "stage": "oft", "model_state": model.state_dict()},
                       os.path.join(ckpt_dir, "oft_pretrained.pt"))
            print("[resume] OFT 完成, 已保存 oft_pretrained.pt", flush=True)
        else:
            print("[resume] OFT 无数据, 跳过", flush=True)
    elif oft_done:
        load_weights(model, os.path.join(ckpt_dir, "oft_pretrained.pt"), device)
        print("[resume] 跳过 OFT (已存在 oft_pretrained.pt)", flush=True)

    # 切回正式 5 类 head
    model.set_num_classes(num_classes, class_counts=cc)
    model.unfreeze_all()

    # ============ LP 阶段 (跳过若已完成) ============
    lp_done = os.path.exists(os.path.join(ckpt_dir, "lp_best.pt"))
    if bool(lp.get("enable", False)) and not lp_done:
        # 论文策略: 冻结前 3 stage, 训 Stage4 + head + STSA + CPAM (Stage4 始终可训)
        print("[resume] LP 阶段: 冻结前3stage, 训 Stage4+head+STSA+CPAM", flush=True)
        model.freeze_first_stages(3)
        train_stage(model, tr_ld, va_ld, cfg, make_criterion(cfg, num_classes, pw, device),
                    class_names, device, num_classes,
                    epochs=int(lp.get("epochs", 5)),
                    base_lr=float(lp.get("lr", 5e-4)), head_mult=1.0,
                    warmup=min(warmup, int(lp.get("epochs", 5))), patience=patience,
                    ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lp")
        print("[resume] LP 完成", flush=True)
    elif lp_done:
        load_weights(model, os.path.join(ckpt_dir, "lp_best.pt"), device)
        print("[resume] 跳过 LP (已存在 lp_best.pt)", flush=True)

    # ============ LP-FT 阶段 (断电续训核心) ============
    resume_ckpt = args.from_ckpt or latest_lpft_ckpt(ckpt_dir)
    ft_epochs = args.epochs or int(lpft.get("epochs", int(cfg["TRAIN"]["epochs"])))
    if args.from_ckpt:
        load_weights(model, args.from_ckpt, device)
        print(f"[resume] 从指定 ckpt 继续 LP-FT, 再训 {ft_epochs} epochs", flush=True)
    elif resume_ckpt and os.path.exists(resume_ckpt):
        load_weights(model, resume_ckpt, device)
        print(f"[resume] 从 {os.path.basename(resume_ckpt)} 继续 LP-FT, 再训 {ft_epochs} epochs", flush=True)
    else:
        print(f"[resume] 无 LP-FT checkpoint, 从头训练 {ft_epochs} epochs", flush=True)

    if bool(lp.get("enable", False)) and lp_done:
        model.freeze_first_stages(3)   # LP 已训过 -> LP-FT 冻结前3 stage
    best = train_stage(model, tr_ld, va_ld, cfg, make_criterion(cfg, num_classes, pw, device),
                       class_names, device, num_classes,
                       epochs=ft_epochs,
                       base_lr=float(lpft.get("lr", float(cfg["TRAIN"]["lr"]))),
                       head_mult=float(lpft.get("head_lr_mult", cfg["TRAIN"].get("head_lr_mult", 10.0))),
                       warmup=warmup, patience=patience,
                       ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lpft")
    print(f"[resume] LP-FT 完成, best f1_macro={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
