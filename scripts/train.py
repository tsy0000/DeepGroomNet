import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet.config import load_config
from deepgroomnet.data import FrameLevelDataset, build_loader, make_pos_weight_from_videos
from deepgroomnet.losses.auxiliary import AuxFocalLoss
from deepgroomnet.models import TAdaConvModel, load_pretrained


CLASSES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]


def read_manifest(path):
    videos = []
    with open(path, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            labels = np.load(row["labels"]) if row["labels"].endswith(".npy") else None
            if labels is None:
                labels = np.array([int(v) for v in row["labels"].split()], dtype=np.int64)
            videos.append({"video": row["video"], "labels": labels.tolist(),
                           "n_frames": len(labels), "mouse": row.get("mouse", Path(row["video"]).stem),
                           "source": row.get("source", "train")})
    return videos


def evaluate(model, loader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            truths.append(y.numpy())
    p = np.concatenate(probs)
    y = np.concatenate(truths)
    scores = []
    for index in range(len(CLASSES)):
        binary = (y == index).astype(int)
        if binary.min() != binary.max():
            scores.append(roc_auc_score(binary, p[:, index]))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Train DeepGroomNet")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out", default="runs/train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["TRAIN"]
    epochs = args.epochs or int(train_cfg["epochs"])
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = read_manifest(args.manifest)
    train_videos = [v for v in videos if v["source"] != "val"]
    val_videos = [v for v in videos if v["source"] == "val"] or train_videos[-1:]
    window = int(cfg["CLIP"]["num_input_frames"])

    train_set = FrameLevelDataset(train_videos, cfg, split="train",
                                  class_names=CLASSES, frame_window=window)
    val_set = FrameLevelDataset(val_videos, cfg, split="val",
                               class_names=CLASSES, frame_window=window)
    train_loader = build_loader(train_set, cfg, split="train",
                                batch_size=int(train_cfg["batch_size"]),
                                num_workers=int(train_cfg["num_workers"]),
                                balanced=True, seed=int(train_cfg["seed"]))
    val_loader = build_loader(val_set, cfg, split="val", batch_size=8,
                              num_workers=0, shuffle=False, balanced=False)

    model = TAdaConvModel(num_classes=len(CLASSES), t_stride=1,
                          use_cpam=bool(cfg["MODEL"]["cpam"]),
                          use_stsa=bool(cfg["MODEL"]["stsa"])).to(device)
    if cfg["MODEL"].get("pretrained"):
        load_pretrained(model, cfg["MODEL"]["pretrained"], device=str(device))

    pos_weight, _ = make_pos_weight_from_videos(train_videos, CLASSES)
    criterion = AuxFocalLoss(len(CLASSES), pos_weight=pos_weight,
                             alpha=float(cfg.get("LOSS", {}).get("alpha", 0.9)),
                             beta=float(cfg.get("LOSS", {}).get("beta", 0.1)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))

    best = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            target = torch.nn.functional.one_hot(y, len(CLASSES)).float().to(device)
            out = model(x)
            if isinstance(out, tuple):
                logits, aux = out
            else:
                logits, aux = out, None
            loss = criterion(logits.float(), target, aux_logits=aux)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {running / step:.4f}", flush=True)
        score = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": running / max(len(train_loader), 1),
                        "val_macro_auroc": score})
        print(f"epoch {epoch}: val_macro_auroc {score:.4f}", flush=True)
        if score >= best:
            best = score
            torch.save({"model_state": model.state_dict(), "cfg": cfg,
                        "classes": CLASSES, "epoch": epoch, "val_macro_auroc": score},
                       out_dir / "best.pt")
        torch.save({"model_state": model.state_dict(), "cfg": cfg, "classes": CLASSES,
                    "epoch": epoch}, out_dir / f"last_{epoch:03d}.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best val macro-AUROC {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
