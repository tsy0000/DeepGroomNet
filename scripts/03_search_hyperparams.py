"""DeepGroomNet_v3 - 浼樺寲鍣?瓒呭弬鎼滅储銆?
用 optuna (若未安装则退化为轻量随机搜索) 搜索训练超参,
鐩爣 = val F1_macro (涓?train.py 鍚屼竴鍚庡鐞嗗彛寰?銆?
鐢ㄦ硶:
  D:\\Anaconda3\\envs\\python310\\python.exe scripts/03_search_hyperparams.py --trials 20
  可选: --config configs/config.yaml --subset-videos 12
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import FrameLevelDataset, make_pos_weight_from_videos
from losses.classification import FocalLossBCE, L2SPRegularizer
from scripts.train import build_model, cosine_with_warmup, make_optimizer

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


def build_dataloaders(cfg, class_names, frame_window, subset_videos, bs, nw):
    split = json.load(open(os.path.join(cfg["DATA"]["meta_root"], "split.json")))
    train_v = split["train"]
    val_v = split["val"]
    if subset_videos:
        train_v = train_v[:subset_videos]
        val_v = val_v[:max(1, subset_videos // 4)]
    tr = FrameLevelDataset(train_v, cfg, split="train",
                           class_names=class_names, frame_window=frame_window)
    va = FrameLevelDataset(val_v, cfg, split="val",
                           class_names=class_names, frame_window=frame_window)
    pw, cc = make_pos_weight_from_videos(train_v, class_names)
    tr_ld = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw,
                       persistent_workers=True, pin_memory=True)
    va_ld = DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw,
                       persistent_workers=True, pin_memory=True)
    return tr_ld, va_ld, pw.to("cpu")


def run_trial(cfg, params, device):
    class_names = cfg["CLASSES"]["openfield"]
    num_classes = len(class_names)
    frame_window = int(cfg["CLIP"]["num_input_frames"])
    bs = int(params["batch_size"])
    nw = 0  # 搜索阶段用 0 worker 省内存
    tr_ld, va_ld, pw = build_dataloaders(cfg, class_names, frame_window,
                                         bs, nw)
    model = build_model(cfg, num_classes).to(device)
    criterion = FocalLossBCE(
        pos_weight=pw.to(device),
        gamma=0.0,
        label_smoothing=float(cfg["LOSS"].get("binary_label_smoothing", 0.05)),
    )
    l2sp = L2SPRegularizer(model, alpha=1e-5, beta=1e-3)
    # 鎼滅储闃舵鐩存帴鍏ㄩ噺璁?(LP-FT) 灏戦噺 epoch
    model.freeze_backbone()
    make_optimizer(model, params["lp_lr"], 1.0, params["weight_decay"])
    # 鐢ㄧ煭 LP 鎵惧埌 head, 鍐?LP-FT
    epochs = max(2, int(params["epochs"]) // 3)
    from scripts.train import train_stage
    model.unfreeze_all()
    opt = make_optimizer(model, params["lr"], params["head_lr_mult"], params["weight_decay"])
    best = 0.0
    for epoch in range(epochs):
        model.train()
        lr_cur = cosine_with_warmup(epoch, epochs, int(params["warmup"]), params["lr"])
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_cur * (params["head_lr_mult"] if gi >= 1 else 1.0)
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            oh = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            loss = criterion(logits.float(), oh) + l2sp(model)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["TRAIN"]["grad_clip"]))
            opt.step()
        P, Y = evaluate_short(model, va_ld, device)
        f1 = f1_macro(P, Y, class_names)
        best = max(best, f1)
    return best


@torch.no_grad()
def evaluate_short(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(device)).float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def f1_macro(P, Y, class_names):
    K = P.shape[1]
    eye = (Y[:, None] == np.arange(K)).astype(int)
    pred = (P >= 0.5).astype(int)
    f1s = []
    for i in range(K):
        tp = int(((eye[:, i] == 1) & (pred[:, i] == 1)).sum())
        fn = int(((eye[:, i] == 1) & (pred[:, i] == 0)).sum())
        fp = int(((eye[:, i] == 0) & (pred[:, i] == 1)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(2 * pre * rec / (pre + rec) if pre + rec else 0.0)
    return float(np.mean(f1s))


def sample_params(trial):
    if trial is None:  # 闅忔満鎼滅储
        import random
        def lp(k): return 10 ** random.uniform(*k)
        return {
            "lr": lp((-5, -3)), "lp_lr": lp((-5, -3)),
            "head_lr_mult": random.uniform(5, 20),
            "weight_decay": lp((-5, -2)),
            "warmup": random.randint(3, 10),
            "epochs": random.randint(8, 20),
            "batch_size": random.choice([8, 16]),
        }
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "lp_lr": trial.suggest_float("lp_lr", 1e-5, 1e-3, log=True),
        "head_lr_mult": trial.suggest_float("head_lr_mult", 5.0, 20.0),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "warmup": trial.suggest_int("warmup", 3, 10),
        "epochs": trial.suggest_int("epochs", 8, 20),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--subset-videos", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.subset_videos:
        cfg["TRAIN"]["search"]["subset_videos"] = args.subset_videos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(ROOT, "outputs", "search")
    os.makedirs(out_dir, exist_ok=True)
    print(f"optuna available: {HAS_OPTUNA}, device: {device}")

    def objective(trial):
        params = sample_params(trial)
        try:
            f1 = run_trial(cfg, params, device)
        except Exception as e:
            print("trial failed:", e)
            return 0.0
        print(f"trial={trial.number if trial is not None else '?'} f1_macro={f1:.3f} params={params}", flush=True)
        return f1

    if HAS_OPTUNA:
        storage = cfg["TRAIN"]["search"].get("storage")
        study = optuna.create_study(direction="maximize",
                                    storage=storage if storage else None,
                                    study_name="deepgroomnet",
                                    load_if_exists=True)
        study.optimize(objective, n_trials=args.trials)
        best = study.best_params
        print("\n=== Best ===")
        print(best)
        print("best f1_macro:", study.best_value)
    else:
        import random
        random.seed(int(cfg["TRAIN"].get("seed", 42)))
        results = []
        for i in range(args.trials):
            params = sample_params(None)
            f1 = run_trial(cfg, params, device)
            results.append((f1, params))
            print(f"trial {i} f1_macro={f1:.3f}", flush=True)
        results.sort(key=lambda x: -x[0])
        print("\n=== Best (random search) ===")
        print("best f1_macro:", results[0][0])
        print(results[0][1])


if __name__ == "__main__":
    main()
