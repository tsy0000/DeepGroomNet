"""DeepGroomNet_v3 - 璁粌涓昏剼鏈€?
瀵归綈 DeepEthogram (甯х骇琛屼负鍒嗙被) + TAdaConv 瀹樻柟璁粌閰嶇疆:
  - 浼樺寲鍣? AdamW, lr=2.5e-4, weight_decay 灏?(TAdaConv 瀹樻柟)
  - 鍒嗗眰 lr: head/鏂板眰 lr脳10 (TAdaConv HEAD_LRMULT=10, 鍥哄畾鍊嶆暟闈炰腑閫旀敼)
  - cosine lr + warmup (TAdaConv 瀹樻柟 8ep warmup)
  - 鎹熷け: pos_weight BCE + label smoothing + L2-SP (loss 鍐?
  - 鏃犻厤棰濋噰鏍峰櫒 (绫诲埆涓嶅钩琛￠潬 pos_weight)
  - 鏃犳瘡 epoch 鍥炴粴 (鏉冮噸鍗曡皟绱Н, 鏍囧噯鍋氭硶)
  - 鎸夊皬榧犲垝鍒? 瀹炴椂甯х骇瑙ｇ爜
"""
import os
import sys
import json
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import numpy as np
import math
import torch
import torch.nn as nn
import yaml

from data.dataset import FrameLevelDataset, make_pos_weight_from_videos
from models.backbones.tadaconvnext import TAdaConvNeXtV2, load_pretrained
from losses.classification import FocalLossBCE, L2SPRegularizer
from postprocess.thresholds import (evaluate_thresholds, threshold_predict,
                                    compute_background, min_bout_per_behavior,
                                    binary_metrics, smooth_probabilities,
                                    hmm_smooth, bout_stats)
from torch.utils.data import DataLoader
from tqdm import tqdm


def binary_class_metrics(eye, pred, class_names):
    """閫愮被浜屽€兼寚鏍?(eye/pred 鍧囦负 [N,K] 0/1)銆傝繑鍥?per_class: acc/precision/recall/f1銆?""
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
    """闃堝€兼棤鍏崇殑 AUROC (杩佺Щ鑷?DeepEthogram metrics.roc_auc / sklearn)銆?
    鐢ㄤ簬鏃╁仠/閫夋ā鍨? 閬垮厤鍦ㄦ瀬灏?val + 绋€鏈夌被涓婂仛闃堝€肩綉鏍兼悳绱㈠鑷?F1 鎶栧姩銆?    P: [N,K] 姒傜巼; Y: [N] int 鏍囩銆?    """
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
    """闃堝€兼棤鍏崇殑 AUPRC (瀵规瀬涓嶅钩琛℃洿绋?銆?""
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
    mc = type("MC", (), {
        "num_classes": num_classes,
        "cpam": cfg["MODEL"].get("cpam", True),
        "cpam_beta": cfg["MODEL"].get("cpam_beta", 0.1),
        "cpam_alpha": cfg["MODEL"].get("cpam_alpha", 0.9),
        "dropout": 0.0,
    })
    model = TAdaConvNeXtV2(mc)
    load_pretrained(model, cfg["MODEL"]["pretrained"])
    model.set_head(num_classes)
    return model


def cosine_with_warmup(epoch, epochs, warmup, lr_base):
    if epoch < warmup:
        return lr_base * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, epochs - warmup)
    return lr_base * 0.5 * (1 + np.cos(np.pi * t))


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]   # (main_logits, aux_logits)
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def save_metrics(path, epoch, class_names, per_class, val_f1m, grad_epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grad_epoch = grad_epoch or {"grad_l2": 0.0, "grad_max": 0.0, "grad_nan": 0}
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "f1_macro", "grad_l2", "grad_max", "grad_nan"]
                       + [f"{c}_acc" for c in class_names]
                       + [f"{c}_p" for c in class_names]
                       + [f"{c}_r" for c in class_names] + [f"{c}_f1" for c in class_names])
        row = [epoch, f"{val_f1m:.2f}", f"{grad_epoch['grad_l2']:.2e}",
               f"{grad_epoch['grad_max']:.2e}", grad_epoch['grad_nan']]
        for c in class_names:
            row += [f"{per_class[c]['accuracy']:.1f}"]
        for c in class_names:
            row += [f"{per_class[c]['precision']:.1f}"]
        for c in class_names:
            row += [f"{per_class[c]['recall']:.1f}"]
        for c in class_names:
            row += [f"{per_class[c]['f1']:.1f}"]
        w.writerow(row)


def make_optimizer(model, base_lr, head_mult, weight_decay):
    head_groups, base_groups = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_groups if name.startswith("head") else base_groups).append(p)
    groups = []
    if base_groups:
        groups.append({"params": base_groups, "lr": base_lr,
                       "weight_decay": weight_decay})
    if head_groups:
        groups.append({"params": head_groups, "lr": base_lr * head_mult,
                       "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def train_stage(model, tr_ld, va_ld, cfg, criterion, l2sp, class_names, device,
                num_classes, epochs, base_lr, head_mult, warmup, patience,
                ckpt_dir, metrics_csv, tag):
    """璺戜竴涓缁冮樁娈?(LP 鎴?LP-FT)銆傝繑鍥炶闃舵鏈€浼?val F1_macro銆?""
    opt = make_optimizer(model, base_lr, head_mult,
                         float(cfg["TRAIN"].get("weight_decay", 0.02)))
    best_f1, best_epoch, bad = 0.0, -1, 0
    for epoch in range(epochs):
        model.train()
        lr_cur = cosine_with_warmup(epoch, epochs, warmup, base_lr)
        # LP 闃舵: head 鏄敮涓€鍙缁?(gi=0), 涓嶄箻 head_mult; LP-FT: head 缁勪箻 head_mult
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_cur * (head_mult if gi >= 1 else 1.0)

        # ---- 璁粌 (tqdm 瀹炴椂鏄剧ず loss/acc/f1/grad) ----
        tr_loss, n_tr = 0.0, 0
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
            data_loss = criterion(logits.float(), oh)
            if aux_logits is not None:
                # CPAM 涓ユ牸鎸夊師鏂囧紡8-10: 鏉′欢鑷€傚簲鍔犳潈 L_CE 涓?L_CPAM (伪=0.9, 尾=0.1)
                lcpam_loss = criterion(aux_logits.float(), oh)
                if lcpam_loss.detach() < data_loss.detach():
                    lce, lc = data_loss.detach(), lcpam_loss.detach()
                    alpha = float(cfg["MODEL"].get("cpam_alpha", 0.9))
                    w_ce = alpha * lce + (1 - alpha) * (lce / (lce + lc + 1e-8))
                    w_cpam = alpha * lc + (1 - alpha) * (lc / (lce + lc + 1e-8))
                    data_loss = w_ce * data_loss + w_cpam * lcpam_loss
                else:
                    beta = float(cfg["MODEL"].get("cpam_beta", 0.1))
                    data_loss = data_loss + beta * lcpam_loss
            loss = data_loss + l2sp(model)
            opt.zero_grad()
            loss.backward()
            # ---- 姊害鍋ュ悍搴﹁缁嗚褰?----
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
            # 瀹炴椂绱鎸囨爣 (鍗曟爣绛?argmax vs 鐪熷€?
            tr_loss += data_loss.item() * x.size(0)
            n_tr += x.size(0)
            grad_l2_sum += grad_l2; grad_max_sum += g_max; grad_nan_sum += g_nan
            ypred = logits.detach().argmax(dim=1)
            for i in range(num_classes):
                t = (y == i); p = (ypred == i)
                tr_tp[i] += ((t & p)).sum().item()
                tr_fp[i] += ((~t) & p).sum().item()
                tr_fn[i] += (t & (~p)).sum().item()
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
        tr_acc = tr_tp.sum() / max(1, tr_n)
        tr_f1s = np.where(tr_tp + tr_fn > 0, 2 * tr_tp / (2 * tr_tp + tr_fp + tr_fn + 1e-9), 0.0)
        # 姣?epoch 姊害姹囨€?        grad_epoch = {"grad_l2": grad_l2_sum / max(1, n_steps),
                      "grad_max": grad_max_sum / max(1, n_steps),
                      "grad_nan": int(grad_nan_sum)}

        # val 璇勪及 (DeepEthogram: 闃堝€肩綉鏍?+ min_bout; 鍙姞鏃跺簭骞虫粦 + bout 缁熻)
        P, Y = evaluate(model, va_ld, cfg, device)
        if cfg["POSTPROCESS"].get("temporal_smooth", False):
            P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
        thresholds, _ = evaluate_thresholds(Y, P, class_names, bg_idx=4)
        pred = threshold_predict(P, thresholds, class_names)
        pred = compute_background(pred, bg_idx=4)
        if cfg["POSTPROCESS"].get("min_bout_per_behavior", False):
            pred = min_bout_per_behavior(pred, Y, class_names,
                                         percentile=cfg["POSTPROCESS"].get("min_bout_percentile", 1),
                                         bg_idx=4)
        eye = (Y[:, None] == np.arange(num_classes)).astype(int)
        per_class = binary_class_metrics(eye, pred, class_names)
        val_f1m = f1_macro_from(per_class, class_names)
        # 闃堝€兼棤鍏崇殑鏃╁仠/閫夋ā鍨嬫寚鏍?(DeepEthogram 鐢?AUROC; 閬垮厤闃堝€肩綉鏍兼姈鍔?
        # select_metric: auroc / auprc / f1
        sel = cfg["TRAIN"].get("select_metric", "auroc")
        if sel == "auprc":
            sel_metric = macro_auprc(P, Y, class_names)
        elif sel == "f1":
            sel_metric = val_f1m
        else:
            sel_metric = macro_auroc(P, Y, class_names)
        save_metrics(metrics_csv, epoch, class_names, per_class, val_f1m, grad_epoch)

        # ---- 姣?epoch 缁撴潫鍚庨€愮被 acc/f1/recall/precision + 姊害姹囨€?----
        print(f"\n[{tag}] ep{epoch+1}/{epochs} | TRAIN loss={tr_loss:.3f} acc={tr_acc*100:.1f} "
              f"f1_macro={tr_f1s.mean()*100:.1f} | VAL f1_macro={val_f1m:.2f} "
              f"AUROC={macro_auroc(P, Y, class_names):.3f} | "
              f"grad_L2={grad_epoch['grad_l2']:.1e} grad_max={grad_epoch['grad_max']:.1e} "
              f"grad_NaN={grad_epoch['grad_nan']}", flush=True)
        print(f"[{tag}] {'琛屼负':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}", flush=True)
        for c in class_names:
            pc = per_class[c]
            print(f"[{tag}] {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%", flush=True)

        torch.save({"epoch": epoch, "stage": tag, "model_state": model.state_dict()},
                   os.path.join(ckpt_dir, f"{tag}_epoch_{epoch:03d}.pt"))
        if sel_metric > best_f1:
            best_f1, best_epoch, bad = sel_metric, epoch, 0
            torch.save({"epoch": epoch, "stage": tag, "model_state": model.state_dict(),
                        "best_metric": best_f1, "val_f1m": val_f1m},
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
                    help="灏嗘帶鍒跺彴杈撳嚭鍚屾椂鍐欏叆璇ユ枃浠?榛樿 outputs/logs/train_<鏃堕棿>.log)")
    args = ap.parse_args()

    # 鑷姩钀界洏鏃ュ織 (PyCharm 杩愯鏃朵篃鑳界敓鎴? 渚夸簬鍚庢湡浼樺寲)
    import time as _t, io as _io
    log_path = args.log_file or os.path.join(
        ROOT, "outputs", "logs", f"train_{_t.strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _logf = open(log_path, "a", encoding="utf-8")
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    class _Tee(_io.TextIOWrapper):
        def __init__(self, orig):
            super().__init__(orig.buffer, encoding=orig.encoding or "utf-8", line_buffering=True)
            self._orig = orig
        def write(self, s):
            try: _logf.write(s); _logf.flush()
            except Exception: pass
            return self._orig.write(s)
        def flush(self):
            self._orig.flush()
    sys.stdout = _Tee(_orig_stdout)
    sys.stderr = _Tee(_orig_stderr)
    print(f"[log] 鎺у埗鍙拌緭鍑哄凡鍐欏叆: {log_path}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    frame_window = int(cfg["CLIP"]["num_input_frames"])

    # 鏁版嵁鍒掑垎 (鎸夊皬榧? 宸茬敱 02 鑴氭湰鐢熸垚)
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    train_v = split["train"]
    val_v = split["val"]
    if args.subset_videos:
        train_v = train_v[:args.subset_videos]
        val_v = val_v[:max(1, args.subset_videos // 4)]
    print(f"train {len(train_v)} 瑙嗛 / val {len(val_v)} 瑙嗛")

    # pos_weight (甯х骇绫诲埆涓嶅钩琛?
    train_ds = FrameLevelDataset(train_v, cfg, split="train",
                                 class_names=class_names, frame_window=frame_window)
    val_ds = FrameLevelDataset(val_v, cfg, split="val",
                               class_names=class_names, frame_window=frame_window)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    print("绫诲埆鍒嗗竷:", dict(zip(class_names, [int(x) for x in cc])))
    print("pos_weight:", [round(x, 2) for x in pw.tolist()])

    # 妯″瀷 + 鎹熷け
    model = build_model(cfg, num_classes).to(device)
    # 绋€鏈夌被鍋忕疆鍒濆鍖? 鐢ㄨ缁冪被鍒崰姣?(DeepEthogram rare_class_bias, 缂撹В鏃╂湡鍏ㄥ垽姝?
    model.set_head(num_classes, class_counts=cc)
    criterion = FocalLossBCE(
        pos_weight=pw.to(device),
        gamma=float(cfg["LOSS"].get("binary_gamma", 0.0)),
        label_smoothing=float(cfg["LOSS"].get("binary_label_smoothing", 0.05)),
        exclusive_pairs=cfg["LOSS"].get("binary_exclusivity_pairs"),
        exclusive_weight=float(cfg["LOSS"].get("binary_exclusivity_weight", 0.0)),
    )
    # L2-SP 姝ｅ垯 (鍙棰勮缁冨眰, 伪)
    from models.backbones.tadaconvnext import load_pretrained as _lp
    l2sp = L2SPRegularizer(model, alpha=1e-5, beta=1e-3)

    bs = int(cfg["TRAIN"]["batch_size"])
    nw = int(cfg["TRAIN"]["num_workers"])
    tr_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                       persistent_workers=True, pin_memory=True)
    va_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                       persistent_workers=True, pin_memory=True)

    warmup = int(cfg["TRAIN"].get("warmup_epochs", 8))
    patience = int(cfg["TRAIN"].get("early_stop_patience", 12))
    metric_dir = cfg["OUTPUT"]["metric_dir"]
    ckpt_dir = cfg["OUTPUT"]["checkpoint_dir"]
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_csv = os.path.join(metric_dir, "train_metrics.csv")
    weight_decay = float(cfg["TRAIN"].get("weight_decay", 0.02))

    # ---- 璁粌绛栫暐: VideoGLUE/LP-FT 涓ゆ寮?(linear probe -> LP-FT) ----
    lp = cfg["TRAIN"].get("lp", {})
    lpft = cfg["TRAIN"].get("lpft", {})
    do_lp = bool(lp.get("enable", False))
    if do_lp:
        # 闃舵1: Linear Probe 鈥?鍐荤粨 backbone, 鍙 head (head_lr_mult=1)
        model.freeze_backbone()
        n_head_params = sum(p.numel() for p in model.head.parameters())
        print(f"[LP] 鍐荤粨 backbone, 鍙 head ({n_head_params} 鍙傛暟)")
        train_stage(
            model, tr_ld, va_ld, cfg, criterion, l2sp, class_names, device,
            num_classes,
            epochs=args.lp_epochs or int(lp.get("epochs", 5)),
            base_lr=float(lp.get("lr", 5e-4)), head_mult=1.0,
            warmup=min(warmup, int(lp.get("epochs", 5))), patience=patience,
            ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lp")

    # 闃舵2: LP-FT 鈥?瑙ｅ喕 backbone, 灏?lr 寰皟 (head_lr_mult=10)
    if do_lp:
        model.unfreeze_all()
    ft_epochs = args.epochs or int(lpft.get("epochs", int(cfg["TRAIN"]["epochs"])))
    best_f1 = train_stage(
        model, tr_ld, va_ld, cfg, criterion, l2sp, class_names, device,
        num_classes,
        epochs=ft_epochs,
        base_lr=float(lpft.get("lr", float(cfg["TRAIN"]["lr"]))),
        head_mult=float(lpft.get("head_lr_mult", cfg["TRAIN"].get("head_lr_mult", 10.0))),
        warmup=warmup, patience=patience,
        ckpt_dir=ckpt_dir, metrics_csv=metrics_csv, tag="lpft")

    print(f"\n瀹屾垚, best val f1_macro={best_f1:.2f}")
    print(f"鎸囨爣 -> {metrics_csv}")

    # ---- 璁粌缁撴潫鍚? 鐢ㄦ渶浣?LP-FT 鏉冮噸鍦ㄧ嫭绔?test 闆嗚瘎浼?----
    test_v = split.get("test") or []
    if test_v:
        print(f"\n[TEST] 鍦?test 闆?{len(test_v)} 瑙嗛)璇勪及鏈€浣虫ā鍨?..")
        test_ds = FrameLevelDataset(test_v, cfg, split="val",
                                    class_names=class_names, frame_window=frame_window)
        test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, persistent_workers=True, pin_memory=True)
        best_ckpt = os.path.join(ckpt_dir, "lpft_best.pt")
        if os.path.exists(best_ckpt):
            ck = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"], strict=False)
            print(f"  杞藉叆 {best_ckpt} (epoch {ck.get('epoch','?')})")
        P, Y = evaluate(model, test_ld, cfg, device)
        if cfg["POSTPROCESS"].get("temporal_smooth", False):
            P = smooth_probabilities(P, int(cfg["POSTPROCESS"].get("smooth_window", 3)))
        thresholds, _ = evaluate_thresholds(Y, P, class_names, bg_idx=4)
        pred = threshold_predict(P, thresholds, class_names)
        pred = compute_background(pred, bg_idx=4)
        if cfg["POSTPROCESS"].get("min_bout_per_behavior", False):
            pred = min_bout_per_behavior(pred, Y, class_names,
                                         percentile=cfg["POSTPROCESS"].get("min_bout_percentile", 1),
                                         bg_idx=4)
        eye = (Y[:, None] == np.arange(num_classes)).astype(int)
        tpc = binary_class_metrics(eye, pred, class_names)
        tf1 = f1_macro_from(tpc, class_names)
        print(f"[TEST] F1_macro={tf1:.2f}")
        print(f"[TEST] {'琛屼负':<12}{'Acc':>7}{'F1':>7}{'Recall':>8}{'Precision':>10}")
        for c in class_names:
            pc = tpc[c]
            print(f"[TEST] {c:<12}{pc['accuracy']:>6.1f}%{pc['f1']:>6.1f}%"
                  f"{pc['recall']:>7.1f}%{pc['precision']:>9.1f}%")
        save_metrics(os.path.join(metric_dir, "test_metrics.csv"), 0, class_names, tpc, tf1)
    else:
        print("\n[TEST] 鏃?test 闆?(split.json 涓?test 涓虹┖)")


if __name__ == "__main__":
    main()
