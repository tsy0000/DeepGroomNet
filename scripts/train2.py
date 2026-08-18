# -*- coding: utf-8 -*-
"""train2.py - 断电续训 (auto-resume).

按已存在的 checkpoint 自动跳过已完成的阶段, 从断点继续:
  1. OFT 阶段:  若 outputs/checkpoints/oft_pretrained.pt 存在 -> 跳过, 否则训练.
  2. LP 阶段:   若 lp_best.pt 存在 -> 跳过, 否则训练.
  3. LP-FT 阶段: 取最新的 lpft_epoch_XXX.pt / lpft_best.pt 加载继续; 否则从头训.

用法:
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/train2.py                # 自动续训
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/train2.py --tag smoke     # 续到 smoke 目录
  D:/Anaconda3/envs/deepgroomnet3/python.exe scripts/train2.py --from-ckpt D:/DeepGroomNet_v3/outputs/checkpoints/lpft_epoch_010.pt --epochs 50
"""
import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

import yaml
import torch

from scripts.train import (build_model, build_oft_videos, train_stage,
                           make_pos_weight_from_videos, cosine_with_warmup)
from data.dataset import FrameLevelDataset
from data.dataloaders import build_loader
from losses.auxiliary import AuxFocalLoss


def make_criterion(cfg, num_classes, pw, device):
    return AuxFocalLoss(
        num_classes=num_classes, pos_weight=pw.to(device),
        gamma=float(cfg["LOSS"].get("binary_gamma", 0.5)),
        alpha=float(cfg["MODEL"].get("cpam_alpha", 0.9)),
        beta=float(cfg["MODEL"].get("cpam_beta", 0.1)),
        l2sp_weight=float(cfg["TRAIN"].get("l2sp_weight", 0.0)),
        label_smoothing=float(cfg["LOSS"].get("binary_label_smoothing", 0.05)))


def load_weights(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"], strict=False)
    print(f"[resume] 加载 {os.path.basename(path)} (epoch={ck.get('epoch','?')} "
          f"stage={ck.get('stage','?')})", flush=True)
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
            print(f"[resume] OFT 预训练: {len(oft_v)} 视频, {len(oft_classes)} 类")
            model.unfreeze_all()
            model.set_num_classes(len(oft_classes))
            oft_pw, _ = make_pos_weight_from_videos(oft_v, oft_classes)
            oft_ds = FrameLevelDataset(oft_v, cfg, split="train", class_names=oft_classes,
                                       frame_window=frame_window)
            oft_ld = build_loader(oft_ds, cfg, split="train", batch_size=bs, num_workers=nw,
                                  balanced=use_balanced, seed=seed)
            # OFT 验证用"小分层验证集" (每类 val_samples_per_class 窗口), 避免对全量 oft_ld 再跑一遍
            oft_val_ds = FrameLevelDataset(oft_v, cfg, split="val", class_names=oft_classes,
                                           frame_window=frame_window)
            oft_val_ld = build_loader(oft_val_ds, cfg, split="val", batch_size=bs,
                                      num_workers=nw, shuffle=False)
            print(f"[resume] OFT 验证集: {len(oft_val_ds)} 样本 (小, 快速)", flush=True)
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
        print("[resume] LP 阶段: 冻结主干, 只训 head+STSA+CPAM", flush=True)
        model.freeze_backbone()
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
