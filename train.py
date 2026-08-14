"""
主训练脚本（模块A + 模块B源头 + 模块C）。

流程：
  1. 读取 config.yaml
  2. 构建 clip 元数据（训练=人工标注段，可选边界抖动）
  3. 构建模型（TAdaConvNeXtV2 + 注意力模块）与 Focal Loss
  4. 渐进式迁移学习：stage1 冻结 backbone 只训 head；stage2 全量微调
  5. 记录类别级 acc/F1 与整体指标

用法：
  python train.py --config configs/config.yaml [--jitter]
"""
import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import ClipDataset
from data.clip_sampler import extract_uniform_clips, homecage_clips, normalize_label_columns
from data import openfield_meta, homecage_meta
from models.backbones.tadaconvnext import build_model
from losses.focal_loss import build_loss, FocalLoss, BinaryFocalLoss
from train_diagnostics import GradTracker, diagnostic_pass
from eval_metrics import (classwise_and_overall, binary_classwise_metrics,
                          save_epoch_metrics_csv, save_confusion_matrix_csv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_clip_rows(meta_csv, cfg):
    """生成 clip 行，返回 (uniform_clips, sliding_clips, pseudo_clips)。

    使用预裁剪 clips.csv。uniform 用于分层划分（train/val/test）；
    sliding（随机滑窗、中心帧标签、可跨边界）仅并入 train——
    实现"训练/推理分割分布一致化"（推理端 MoSeq 音节窗口即含边界的窗口）。
    pseudo（LLM 语义校验的高置信伪标签，scripts/08_llm_self_train.py 生成）
    仅并入 train（绝不分入 val/test，防止伪标签污染评估）。

    发表级要求：clips.csv 必须覆盖 meta 中全部小鼠，否则视为不完整，拒绝使用
    （防止"部分预裁剪 + 静默降级"造成实验数据缺失）。
    """
    import csv as _csv

    clips_csv = os.path.join(cfg["DATA"]["clip_output_root"], "clips.csv")
    if not (cfg["DATA"].get("use_pre_extracted", True) and os.path.exists(clips_csv)):
        raise RuntimeError("必须先生成预裁剪 clips.csv（python scripts/02_gen_clips.py）。")

    rows = []
    with open(clips_csv, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            r.setdefault("source", "uniform")
            r.setdefault("video", r["clip"])
            rows.append(r)

    has_sliding = any(r.get("source") == "sliding" for r in rows)
    if not has_sliding:
        raise RuntimeError(
            "clips.csv 缺少 sliding 行（source 列缺失或全为 uniform）。"
            "sliding clips 是训练/推理一致化（模块B）的必需组件。"
            "请运行：python scripts/02_gen_clips.py --overwrite 重新生成。")

    meta_mice = set()
    with open(meta_csv, encoding="utf-8") as f:
        for m in _csv.DictReader(f):
            meta_mice.add(m["mouse"])
    clip_mice = set()
    for r in rows:
        clip_mice.add(os.path.basename(r["clip"]).split("_")[0])
    missing = meta_mice - clip_mice
    if missing:
        raise RuntimeError(
            f"clips.csv 不完整：缺少 {len(missing)} 只小鼠 {sorted(missing)}。"
            f"请先运行 scripts/02_gen_clips.py 完成全量预裁剪（当前仅 {len(clip_mice)}/{len(meta_mice)} 只）。"
            f"不要使用不完整的 clips.csv 训练——这会静默改变实验数据。")

    uniform = [r for r in rows if r["source"] not in ("sliding", "pseudo")]
    sliding = [r for r in rows if r["source"] == "sliding"]

    pseudo = []
    pseudo_csv = os.path.join(cfg["DATA"]["clip_output_root"], "clips_pseudo.csv")
    if cfg.get("SELFTRAIN", {}).get("enabled", False) and os.path.exists(pseudo_csv):
        with open(pseudo_csv, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if not r.get("video") or not r.get("start") or not r.get("label"):
                    continue
                pseudo.append({"video": r["video"], "start": int(r["start"]),
                               "end": int(r["end"]), "label": r["label"],
                               "source": "pseudo"})
        if pseudo:
            print(f"[pseudo] 并入训练集伪标签 {len(pseudo)} 条"
                  f"（仅 train，不进 val/test）")
    return uniform, sliding, pseudo


def _mouse_key(c):
    """小鼠键：预裁剪 clip 取文件名前缀；原始视频取父目录名。"""
    if "clip" in c and c["clip"]:
        return os.path.basename(c["clip"]).split("_")[0]
    return os.path.basename(os.path.dirname(c["video"]))


def split_clips(clips, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按小鼠个体分层划分 train/val/test（发表标准）。

    贪心配额分配：
      目标 clip 数 = [0.70, val_ratio, test_ratio] * N；
      每只小鼠放入「KL(全局比例, 放入后比例) + 配额超额重罚」最小的 split；
      小鼠顺序由 seed 打乱，保证可复现；配额只软约束（最后几只小鼠允许少量溢出）。
    """
    import numpy as np
    rng = random.Random(seed)
    by_mouse = {}
    for c in clips:
        by_mouse.setdefault(_mouse_key(c), []).append(c)
    mice = sorted(by_mouse.keys())
    rng.shuffle(mice)

    class_names = sorted({c["label"] for c in clips})
    idx = {n: i for i, n in enumerate(class_names)}
    K = len(class_names)
    global_counts = np.zeros(K, dtype=np.float64)
    for c in clips:
        global_counts[idx[c["label"]]] += 1
    global_prop = global_counts / global_counts.sum()
    N = len(clips)
    targets = {"train": (1 - val_ratio - test_ratio) * N,
               "val": val_ratio * N,
               "test": test_ratio * N}

    splits = {"train": [], "val": [], "test": []}
    cur = {k: np.zeros(K, dtype=np.float64) for k in splits}
    PEN = 1e6

    def kl(a, b):
        eps = 1e-9
        a = a + eps
        b = b + eps
        return float(np.sum(a * np.log(a / b)))

    def score(k, mc):
        cand = cur[k] + mc
        prop = cand / cand.sum()
        over = max(0.0, cand.sum() - targets[k])
        return kl(global_prop, prop) + PEN * over

    for m in mice:
        mc = np.zeros(K, dtype=np.float64)
        for c in by_mouse[m]:
            mc[idx[c["label"]]] += 1
        best = min(splits.keys(), key=lambda k: score(k, mc))
        splits[best].extend(by_mouse[m])
        cur[best] += mc

    return splits["train"], splits["val"], splits["test"]


def _apply_criterion(criterion, logits, targets, class_counts):
    """兼容 FocalLoss(需要 class_counts)、BinaryFocalLoss(one-hot) 与标准 CrossEntropyLoss。"""
    if isinstance(criterion, FocalLoss):
        return criterion(logits, targets, class_counts)
    if isinstance(criterion, BinaryFocalLoss):
        oh = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).float()
        return criterion(logits, oh)
    return criterion(logits, targets)


def dataset_cls_ids(dataset):
    """返回每个样本的类别 id（用于加权采样）。"""
    ids = []
    for i in range(len(dataset)):
        c = dataset.clips[i]
        ids.append(dataset.class_to_idx(c["label"]))
    return ids


def make_train_loader(dataset, cfg, class_counts=None):
    """带类别平衡采样的训练 DataLoader（主流做法，缓解极端不平衡）。

    sampler_weighted=True 时用反频 WeightedRandomSampler（稀有类高权重）。
    另支持 Sakamoto 2022 式每 epoch 配额采样（sampler_quota）：
      - 每 epoch 稀有类最多抽 quota 段（FG/BG 限频，防 59x 过采样"背数据"）
      - 常见类按占比填充余量（背景/Locomote/Rear）
    参考 Sakamoto：每 epoch 1000 背景/200 FG/400 BG（bg 与他相反——他 FG 最少）。
    """
    if class_counts is not None and cfg["TRAIN"].get("sampler_quota", False):
        return _make_quota_loader(dataset, cfg, class_counts)
    if class_counts is not None and cfg["TRAIN"].get("sampler_weighted", True):
        counts = class_counts.float()
        weights = (1.0 / counts).clamp(max=1e3)
        ids = dataset_cls_ids(dataset)
        w = weights[ids].tolist()
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True
    return DataLoader(dataset, batch_size=cfg["TRAIN"]["batch_size"],
                      shuffle=shuffle, sampler=sampler,
                      num_workers=int(cfg["TRAIN"].get("num_workers", 2)),
                      pin_memory=bool(cfg["TRAIN"].get("pin_memory", False)),
                      persistent_workers=bool(cfg["TRAIN"].get("persistent_workers", False)))


def _make_quota_loader(dataset, cfg, class_counts):
    """Sakamoto 2022 式每 epoch 配额采样：稀有类限频 + 常见类按占比填充。

    每 epoch 生成固定长度的采样序列（batch 数 = quota_total / batch_size），
    稀有类（配额类）以固定上限抽取（可重复），其余由常见类按反频权重填满。
    防"WeightedRandomSampler 59x 过采样 → 模型背下稀有类样本而非泛化"。
    """
    import numpy as _np
    from collections import Counter as _Counter
    ids = dataset_cls_ids(dataset)
    id_counts = _Counter(ids)
    total = len(ids)
    n_classes = int(class_counts.shape[0])
    default_quota = {2: 200, 3: 400}
    quota = {int(k): int(v) for k, v in cfg["TRAIN"].get("sampler_quota_map", {}).items()}
    default_quota.update(quota)
    budget = max(total, sum(min(default_quota.get(c, 1), id_counts[c])
                            for c in default_quota if c in id_counts))
    rng = _np.random.default_rng(cfg["TRAIN"].get("seed", 42))
    sample_ids = []
    common = [c for c in range(n_classes) if c not in default_quota]
    for c, n in default_quota.items():
        if c in id_counts:
            pool = [i for i, x in enumerate(ids) if x == c]
            take = min(n, len(pool))
            sample_ids.extend(rng.choice(pool, take, replace=True).tolist())
    common_weights = _np.array([1.0 / max(float(class_counts[c]), 1e-6)
                                for c in common], dtype=float)
    common_weights /= common_weights.sum()
    remain = budget - len(sample_ids)
    if remain > 0:
        pools = {c: [i for i, x in enumerate(ids) if x == c] for c in common}
        while len(sample_ids) < budget:
            c = int(rng.choice(common, p=common_weights))
            if pools[c]:
                sample_ids.append(int(rng.choice(pools[c])))
    rng.shuffle(sample_ids)
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SubsetRandomSampler(sample_ids),
        batch_size=cfg["TRAIN"]["batch_size"], drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler,
                      num_workers=int(cfg["TRAIN"].get("num_workers", 2)),
                      pin_memory=bool(cfg["TRAIN"].get("pin_memory", False)),
                      persistent_workers=bool(cfg["TRAIN"].get("persistent_workers", False)))


def train_epoch(model, loader, criterion, opt, device, class_counts, scaler=None,
                grad_accum=1, aux_loader=None, aux_criterion=None, aux_weight=0.0,
                desc="train", sem_mat=None, sem_weight=0.0, sem_aux_mat=None,
                grad_tracker=None, self_grad_clip=0.0, aux_every=1, l2sp=None):
    """梯度累积：有效 batch = batch_size * grad_accum（batch=2 硬件限制下的稳定训练）。

    aux_loader: 多任务辅助数据（homecage 8 类，弥补 openfield 数据量少）。
    aux_weight: 辅助损失权重 λ（L = L_main + λ·L_aux）。
    sem_mat/sem_aux_mat: LLM 语义中介（D-4）目标相似度矩阵，加到主头/辅助头权重行
    （sem_weight * MSE(cos_sim(W_head) - S_LLM)，见 llm/semantic_prior.py）。
    grad_tracker: 训练诊断（train_diagnostics.GradTracker），每 step 记录分组梯度范数。
    self_grad_clip: 全局梯度裁剪范数（0=关闭；SlowFast/MMAction2 标准做法）。
    """
    from llm.semantic_prior import semantic_regularization_loss
    from tqdm import tqdm
    model.train()
    total, correct = 0, 0
    running_loss = 0.0
    all_p, all_t = [], []
    aux_iter = iter(aux_loader) if aux_loader is not None else None
    opt.zero_grad()
    pbar = tqdm(loader, desc=desc, unit="batch", ncols=110,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for step, (x, pose, kp_map, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        pose = pose.to(device) if pose is not None else None
        kp_map = kp_map.to(device) if kp_map is not None else None
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x, pose=pose, kp_map=kp_map)
        loss = _apply_criterion(criterion, logits.float(), y, class_counts)
        if l2sp is not None:
            loss = loss + l2sp(model)
        if (sem_mat is not None and sem_weight > 0
                and sem_mat.shape[0] == model.head[3].weight.shape[0]):
            loss = loss + sem_weight * semantic_regularization_loss(
                model.head[3].weight, sem_mat)
        loss = loss / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if aux_iter is not None and aux_weight > 0 and step % aux_every == 0:
            try:
                xa, _pa, _ka, ya = next(aux_iter)
            except StopIteration:
                aux_iter = iter(aux_loader)
                xa, _pa, _ka, ya = next(aux_iter)
            xa, ya = xa.to(device), ya.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                _, aux_logits = model(xa, return_aux=True)
            loss_aux = _apply_criterion(aux_criterion, aux_logits.float(), ya, None)
            if (sem_aux_mat is not None and sem_weight > 0
                    and sem_aux_mat.shape[0] == model.aux_head[3].weight.shape[0]):
                loss_aux = loss_aux + sem_weight * semantic_regularization_loss(
                    model.aux_head[3].weight, sem_aux_mat)
            loss_aux = loss_aux / grad_accum
            if scaler is not None:
                scaler.scale(loss_aux).backward()
            else:
                loss_aux.backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(opt)
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                print(f"  [warn] step {step} 梯度 NaN/Inf: {bad[:3]} "
                      f"scale={scaler.get_scale():.0f}（step 自动跳过+降 scale）",
                      flush=True)
            if self_grad_clip > 0:
                pre_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.grad is not None],
                    self_grad_clip)
                if step % 800 == 399:
                    _sc = f" scale={scaler.get_scale():.0f}" if scaler is not None else ""
                    print(f"  [clip] pre={pre_norm:.3e} clip={self_grad_clip}{_sc}",
                          flush=True)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad()
        if grad_tracker is not None:
            grad_tracker.update()
        running_loss += loss.item() * y.size(0) * grad_accum
        pred = logits.float().argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_p.extend(pred.cpu().tolist())
        all_t.extend(y.cpu().tolist())
        pbar.set_postfix_str(f"loss={running_loss / max(total, 1):.3f} "
                             f"acc={correct / max(total, 1) * 100:.1f}%")
    acc = accuracy_score(all_t, all_p) * 100
    f1 = f1_score(all_t, all_p, average="macro", zero_division=0) * 100
    grad_summary = grad_tracker.summary() if grad_tracker is not None else None
    return running_loss / total, acc, f1, grad_summary


@torch.no_grad()
def collect_probs(model, loader, device):
    """收集 sigmoid 概率与标签（DeepEthogram 式阈值调优输入）。"""
    from tqdm import tqdm
    model.eval()
    all_p, all_t = [], []
    pbar = tqdm(loader, desc="collect-probs", unit="batch", ncols=100,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for x, pose, kp_map, y in pbar:
        x = x.to(device)
        pose = pose.to(device) if pose is not None else None
        kp_map = kp_map.to(device) if kp_map is not None else None
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x, pose=pose, kp_map=kp_map).float()
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_t.extend(y.tolist())
    return np.concatenate(all_p), np.array(all_t)


def eval_epoch(model, loader, criterion, device, class_names, epoch, split, metrics_csv, cm_dir,
               desc="eval"):
    from tqdm import tqdm
    model.eval()
    binary = isinstance(criterion, BinaryFocalLoss)
    total, correct = 0, 0
    running_loss = 0.0
    all_p, all_t, all_logits = [], [], []
    pbar = tqdm(loader, desc=desc, unit="batch", ncols=110,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for x, pose, kp_map, y in pbar:
        x, y = x.to(device), y.to(device)
        pose = pose.to(device) if pose is not None else None
        kp_map = kp_map.to(device) if kp_map is not None else None
        logits = model(x, pose=pose, kp_map=kp_map)
        loss = _apply_criterion(criterion, logits, y, None)
        running_loss += loss.item() * y.size(0)
        all_logits.append(logits.float().detach().cpu().numpy())
        pred = logits.float().argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_p.extend(pred.cpu().tolist())
        all_t.extend(y.cpu().tolist())
        pbar.set_postfix_str(f"loss={running_loss / max(total, 1):.3f} "
                             f"acc={correct / max(total, 1) * 100:.1f}%")
    if binary:
        import numpy as _np
        from losses.focal_loss import FocalLoss as _FL
        probs = 1.0 / (1.0 + _np.exp(-_np.concatenate(all_logits)))
        overall, per_class, cm = binary_classwise_metrics(
            all_t, probs, class_names,
            threshold=float(getattr(criterion, "_eval_threshold", 0.5)))
    else:
        overall, per_class, cm = classwise_and_overall(all_t, all_p, class_names)
    if metrics_csv:
        save_epoch_metrics_csv(metrics_csv, epoch, split, class_names, overall, per_class)
    if cm_dir and cm is not None:
        os.makedirs(cm_dir, exist_ok=True)
        save_confusion_matrix_csv(os.path.join(cm_dir, f"{split}_epoch_{epoch:03d}_cm.csv"),
                                  cm, class_names, split, epoch)
    return (running_loss / total, overall, per_class, cm)


def load_homecage_clips(cfg):
    """构建 homecage 8类中间域 clip 元数据（整段切窗）。"""
    import csv as _csv
    meta_csv = os.path.join(cfg["DATA"]["meta_root"], "homecage_meta.csv")
    if not os.path.exists(meta_csv):
        homecage_meta.build_homecage_meta(cfg["DATA"]["homecage_root"], meta_csv)
    rows = []
    with open(meta_csv, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    clips = []
    hc_classes = cfg["CLASSES"]["homecage"]
    for r in rows:
        if r["class"] not in hc_classes:
            continue
        for (s, e) in homecage_clips(r["clip"], cfg, hc_classes):
            clips.append({"video": r["clip"], "start": s, "end": e, "label": r["class"]})
    return clips, hc_classes


class HomecageDataset(ClipDataset):
    """homecage 8类：label 映射到 homecage 类名。"""

    def class_to_idx(self, label):
        return self.hc_classes.index(label)

    def __init__(self, clips, cfg, split="train", hc_classes=None):
        self.hc_classes = hc_classes or cfg["CLASSES"]["homecage"]
        super().__init__(clips, cfg, split)


def run_stage(model, name, tr_ld, va_ld, cfg, criterion, device, class_names,
              lr, epochs, freeze_first, metric_dir, patience=10, resume_path=None,
              aux_loader=None, aux_criterion=None, aux_weight=0.0,
              sem_mat=None, sem_weight=0.0, sem_aux_mat=None, l2sp=None):
    """通用阶段训练（stage1 冻结只训 head；stage2 全量；可选多任务辅助训练）。
    每个阶段的指标/混淆矩阵写入独立文件（避免阶段间 epoch 编号冲突，发表级可追溯）。
    每 epoch 保存 last_{name}.pt；resume_path 指向该文件时从断点继续（--resume）。"""
    stage_metrics = os.path.join(metric_dir, f"epoch_metrics_{name}.csv")
    stage_cm = os.path.join(metric_dir, "confusion_matrices", name)
    ckpt_dir = cfg["OUTPUT"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, f"last_{name}.pt")

    if freeze_first:
        model.freeze_backbone()
    else:
        model.unfreeze_all()
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=cfg["TRAIN"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["TRAIN"].get("mixed_precision", True))
                                       and torch.cuda.is_available())
    grad_tracker = (GradTracker(model)
                    if cfg["TRAIN"].get("diagnostics", True) else None)
    diag_every = int(cfg["TRAIN"].get("diagnostics_every", 1))
    diag_batches = int(cfg["TRAIN"].get("diagnostics_val_batches", 8))

    best_f1, best_state, best_epoch, bad, start_epoch = 0.0, None, 0, 0, 0
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        r = model.load_state_dict(ck["model_state"], strict=False)
        if r.missing_keys or r.unexpected_keys:
            print(f"[{name}] resume 键差异: missing={len(r.missing_keys)} "
                  f"unexpected={len(r.unexpected_keys)}（头部尺寸变化属正常）")
        opt.load_state_dict(ck["optimizer_state"])
        if "scaler_state" in ck and ck["scaler_state"] is not None and scaler is not None:
            scaler.load_state_dict(ck["scaler_state"])
        best_f1, best_epoch, bad, start_epoch = (ck["best_f1"], ck["best_epoch"],
                                                 ck["bad"], ck["epoch"] + 1)
        print(f"[{name}] resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        import time as _time
        _t0 = _time.time()
        lr_cur = _cosine_lr(epoch, cfg, epochs) if not freeze_first else lr
        for g in opt.param_groups:
            g["lr"] = lr_cur
        tr_loss, tr_acc, tr_f1, grad_summ = train_epoch(
            model, tr_ld, criterion, opt, device,
            class_counts=None, scaler=scaler,
            grad_accum=int(cfg["TRAIN"].get("grad_accum", 1)),
            aux_loader=aux_loader,
            aux_criterion=aux_criterion,
            aux_weight=aux_weight,
            desc=f"[{name}] ep{epoch + 1}/{epochs} train",
            sem_mat=sem_mat, sem_weight=sem_weight,
            sem_aux_mat=sem_aux_mat,
            grad_tracker=grad_tracker,
            self_grad_clip=(float(cfg["TRAIN"].get("grad_clip_headonly", 50.0))
                            if freeze_first else float(cfg["TRAIN"].get("grad_clip", 5.0))),
            aux_every=int(cfg["TRAIN"].get("aux_every", 1)),
            l2sp=l2sp)
        vloss, voverall, vper, _ = eval_epoch(model, va_ld, criterion, device, class_names,
                                              epoch, "val", stage_metrics, stage_cm,
                                              desc=f"[{name}] ep{epoch + 1}/{epochs} val")
        _el = _time.time() - _t0
        _eta = _el * (epochs - epoch - 1) / max(epoch - start_epoch + 1, 1)
        cls_str = " ".join(f"{k} r{v['recall']:.1f}/P{v['precision']:.1f}/F{v['f1']:.1f}"
                           for k, v in vper.items())
        diag_str = ""
        if grad_tracker is not None and grad_summ is not None:
            g = grad_summ
            diag_str = (f" | grad ds={g['downsample']:.2e} st={g['stages']:.2e} "
                        f"att={g['attention']:.2e} hd={g['head']:.2e}")
        print(f"  [{name}] ep{epoch+1}/{epochs} tr_acc={tr_acc:.2f} tr_f1={tr_f1:.2f} "
              f"val_acc={voverall['acc']:.2f} "
              f"val_f1m={voverall['f1_macro']:.2f} | {cls_str} | {_el:.0f}s/ep | "
              f"ETA {_eta / 60:.0f}min{diag_str}")
        if grad_tracker is not None and (epoch % diag_every == diag_every - 1
                                         or epoch == epochs - 1):
            gate_rows, logit_rows, warn, gate_by_class = diagnostic_pass(
                model, va_ld, cfg, class_names, metric_dir,
                tag=f"{name}_ep{epoch + 1:03d}", num_batches=diag_batches)
            g_str = " | ".join(f"st{i}gate={s['mean']:.2f}(sat{s['sat']:.2f}g{s['conc']:.3f})"
                               for i, s in sorted({x["stage"]: x for x in gate_rows}.items()))
            c_str = " | ".join(f"st{i}类:{','.join(f'{k}={v:.2f}' for k, v in bc.items())}"
                               for i, bc in gate_by_class.items())
            l_str = " | ".join(f"{r['class']} lg={r['mean_logit']:.2f} "
                               f"p={r['prob']:.2f} sep={r['sep']:.2f}" for r in logit_rows)
            print(f"     [diag] {g_str}\n     [diag-class] {c_str}\n     [diag-logit] {l_str}")
            for w in warn:
                print(f"     [diag!] {w}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad},
                   last_path)
        torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad},
                   os.path.join(ckpt_dir, f"epoch_{name}_ep{epoch + 1:03d}.pt"))
        if voverall["f1_macro"] > best_f1:
            best_f1, best_epoch, bad = voverall["f1_macro"], epoch + 1, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1, best_state, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--skip_homecage", action="store_true", help="跳过 homecage 中间域阶段")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_epochs", type=int, default=None, help="覆盖 S2 最大 epoch（调试用）")
    ap.add_argument("--subset", type=int, default=None, help="仅用前 N 个训练 clip（冒烟测试）")
    ap.add_argument("--resume", default=None, help="从 last_openfield-s2.pt 断点续训")
    ap.add_argument("--attention", default=None,
                    help="覆盖 MODEL.attention（A/B 对照：none/motion/part/motion+part/motion_stat）")
    ap.add_argument("--no-semantic", action="store_true",
                    help="关闭 LOSS.semantic_prior（A/B 对照：EthoCLIP 语义正则开关）")
    ap.add_argument("--no-mask-prior", action="store_true",
                    help="关闭 MODEL.attention_mask_prior（A/B 对照：小鼠掩码先验开关）")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="关闭训练诊断（梯度范数/注意力门统计/热图）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.attention is not None:
        cfg["MODEL"]["attention"] = args.attention
    if args.no_semantic:
        cfg["LOSS"]["semantic_prior"] = False
    if args.no_diagnostics:
        cfg["TRAIN"]["diagnostics"] = False
    if args.no_mask_prior:
        cfg["MODEL"]["attention_mask_prior"] = 0.0
    if args.seed is not None:
        cfg["TRAIN"]["seed"] = args.seed
    set_seed(cfg["TRAIN"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    metric_dir = cfg["OUTPUT"]["metric_dir"]
    os.makedirs(metric_dir, exist_ok=True)

    class_names = list(cfg["CLASSES"]["openfield_keep"])
    meta_csv = os.path.join(cfg["DATA"]["meta_root"], "openfield_meta.csv")
    if not os.path.exists(meta_csv):
        openfield_meta.build_openfield_meta(cfg["DATA"]["openfield_roots"], meta_csv)
    uniform_clips, sliding_clips, pseudo_clips = build_clip_rows(meta_csv, cfg)
    tr, va, te = split_clips(uniform_clips,
                             val_ratio=cfg["TRAIN"]["split"]["val_ratio"],
                             test_ratio=cfg["TRAIN"]["split"]["test_ratio"],
                             seed=cfg["TRAIN"]["seed"])
    tr = tr + sliding_clips
    tr = tr + pseudo_clips
    print(f"[openfield] uniform={len(uniform_clips)} sliding={len(sliding_clips)} "
          f"pseudo={len(pseudo_clips)} "
          f"train={len(tr)} val={len(va)} test={len(te)} "
          f"(按小鼠分层 70/15/15, seed={cfg['TRAIN']['seed']})")
    if args.subset:
        rng = random.Random(cfg["TRAIN"]["seed"])
        tr = rng.sample(tr, min(args.subset, len(tr)))
        print(f"   [smoke] 训练 clip 子集 -> {len(tr)}")
    from collections import Counter
    print("   训练集类别分布:", dict(Counter(c["label"] for c in tr)))
    print("   验证集类别分布:", dict(Counter(c["label"] for c in va)))
    print("   测试集类别分布:", dict(Counter(c["label"] for c in te)))

    class_counts = torch.tensor([sum(1 for c in tr if c["label"] == n) for n in class_names],
                                dtype=torch.float32).clamp(min=1)
    criterion = build_loss(cfg["LOSS"], class_counts)
    if isinstance(criterion, (FocalLoss, BinaryFocalLoss)) and criterion.alpha_values is not None:
        alpha_log = {n: round(a, 4) for n, a in zip(class_names, criterion.alpha_values)}
        print("  [loss] 类别权重 alpha:", alpha_log)

    model = build_model(cfg).to(device)
    from losses.l2sp import build_l2sp
    l2sp = build_l2sp(cfg["LOSS"], model)
    if l2sp is not None:
        print(f"[l2sp] L2-SP 防预训练遗忘开启（α={l2sp.alpha}, β={l2sp.beta}, "
              f"覆盖 {len(l2sp._names)} 个预训练参数张量）")

    sem_mat = sem_aux_mat = None
    sem_weight = float(cfg["LOSS"].get("semantic_weight", 0.0))
    if cfg["LOSS"].get("semantic_prior", False) and sem_weight > 0:
        from llm.semantic_prior import (llm_semantic_matrices, enhance_semantic_matrix,
                                        reorder_matrix,
                                        TYPE_LEVELS_OPENFIELD, TYPE_LEVELS_HOMECAGE,
                                        TYPE_GROUP_OPENFIELD, TYPE_GROUP_HOMECAGE,
                                        OPENFIELD_DEFS, HOMECAGE_DEFS)
        sem_cache = os.path.join(cfg["DATA"]["meta_root"], "semantic_matrices.json")
        smats = llm_semantic_matrices(cfg, sem_cache)
        if cfg["LOSS"].get("semantic_ose", False) or cfg["LOSS"].get("semantic_oagm", False):
            smats["openfield"] = enhance_semantic_matrix(
                smats["openfield"], list(OPENFIELD_DEFS.keys()), TYPE_LEVELS_OPENFIELD,
                TYPE_GROUP_OPENFIELD,
                use_ose=bool(cfg["LOSS"].get("semantic_ose", False)),
                ose_alpha=float(cfg["LOSS"].get("semantic_ose_alpha", 0.5)),
                use_oagm=bool(cfg["LOSS"].get("semantic_oagm", False)),
                lam=float(cfg["LOSS"].get("semantic_oagm_lambda", 2.0)),
                hops=int(cfg["LOSS"].get("semantic_oagm_hops", 2)),
                tau=float(cfg["LOSS"].get("semantic_oagm_tau", 0.5)))
            smats["homecage"] = enhance_semantic_matrix(
                smats["homecage"], list(HOMECAGE_DEFS.keys()), TYPE_LEVELS_HOMECAGE,
                TYPE_GROUP_HOMECAGE,
                use_ose=bool(cfg["LOSS"].get("semantic_ose", False)),
                ose_alpha=float(cfg["LOSS"].get("semantic_ose_alpha", 0.5)),
                use_oagm=bool(cfg["LOSS"].get("semantic_oagm", False)),
                lam=float(cfg["LOSS"].get("semantic_oagm_lambda", 2.0)),
                hops=int(cfg["LOSS"].get("semantic_oagm_hops", 2)),
                tau=float(cfg["LOSS"].get("semantic_oagm_tau", 0.5)))
        sem_mat = reorder_matrix(smats["openfield"], list(OPENFIELD_DEFS.keys()), class_names)
        sem_aux_mat = reorder_matrix(smats["homecage"], list(HOMECAGE_DEFS.keys()),
                                     list(cfg["CLASSES"]["homecage"]))
        print(f"[semantic_prior] 语义正则开启 λ={sem_weight}"
              f"（openfield {sem_mat.shape[0]}×{sem_mat.shape[1]}, "
              f"homecage {sem_aux_mat.shape[0]}×{sem_aux_mat.shape[1]}）"
              f" OSE={cfg['LOSS'].get('semantic_ose', False)}"
              f" OAGM={cfg['LOSS'].get('semantic_oagm', False)}")

    hc_tr_ld = hc_va_ld = hc_crit = None
    if not args.skip_homecage:
        hc_clips, hc_classes = load_homecage_clips(cfg)
        if hc_clips:
            rng = random.Random(cfg["TRAIN"]["seed"])
            hc_videos = sorted(set(c["video"] for c in hc_clips))
            rng.shuffle(hc_videos)
            n_val = max(1, int(len(hc_videos) * 0.15))
            hc_val_v = set(hc_videos[:n_val])
            hc_tr = [c for c in hc_clips if c["video"] not in hc_val_v]
            hc_va = [c for c in hc_clips if c["video"] in hc_val_v]
            hc_tr_ds = HomecageDataset(hc_tr, cfg, "train", hc_classes)
            hc_va_ds = HomecageDataset(hc_va, cfg, "val", hc_classes)
            hc_counts = torch.tensor([sum(1 for c in hc_tr if c["label"] == n)
                                      for n in hc_classes], dtype=torch.float32).clamp(min=1)
            hc_tr_ld = make_train_loader(hc_tr_ds, cfg, hc_counts)
            hc_va_ld = DataLoader(hc_va_ds, batch_size=cfg["TRAIN"]["batch_size"],
                                  shuffle=False, num_workers=cfg["TRAIN"]["num_workers"],
                                  pin_memory=cfg["TRAIN"].get("pin_memory", False))
            model.set_head(len(hc_classes))
            if cfg["LOSS"].get("rare_class_bias_init", False):
                FocalLoss.rare_class_bias(model.head[3], hc_counts)
                print("  [loss] homecage head 稀有类偏置初始化 b=-log((1-π)/π)")
            print(f"[homecage] 中间域: train={len(hc_tr)} val={len(hc_va)} (8类)")
            hc_crit = build_loss(cfg["LOSS"], hc_counts)
            hc_epochs = int(cfg["TRAIN"]["transfer_stages"].get("homecage_epochs", 0))
            if hc_epochs > 0:
                run_stage(model, "homecage", hc_tr_ld, hc_va_ld, cfg, hc_crit, device,
                          hc_classes, lr=cfg["TRAIN"]["transfer_stages"]["stage1_lr"],
                          epochs=hc_epochs,
                          freeze_first=True, metric_dir=metric_dir,
                          sem_mat=sem_aux_mat, sem_weight=sem_weight, l2sp=l2sp)
            if cfg["TRAIN"].get("aux_homecage", False):
                model.set_aux_head(len(hc_classes))
                print(f"[aux] homecage 辅助头已初始化（8类，λ={cfg['TRAIN'].get('aux_weight', 0.3)}）")

    model.set_head(len(class_names))
    if cfg["LOSS"].get("rare_class_bias_init", False):
        FocalLoss.rare_class_bias(model.head[3], class_counts)
        print("  [loss] openfield head 稀有类偏置初始化 b=-log((1-π)/π)")
    if cfg.get("KEYPOINTS", {}).get("enabled", True) and cfg.get("KEYPOINTS", {}).get("root"):
        kp_parts = cfg["KEYPOINTS"].get("use_parts") or None
        kp_k = len(kp_parts) if kp_parts else 27
        from data.pose_target import SUPERANIMAL_SKELETONS
        kp_c = kp_k + (len(SUPERANIMAL_SKELETONS) if kp_parts is None else 0)
        hm = int(cfg["KEYPOINTS"].get("render_heatmap_size", 64))
        model.set_pose_branch(len(class_names), in_channels=kp_c, heatmap_size=hm)
        print(f"[keypoints] PoseC3D 热图分支已初始化（{kp_c} 通道 × {hm}×{hm} 热图 → 3D CNN）")
    tr_ds = ClipDataset(tr, cfg, split="train")
    va_ds = ClipDataset(va, cfg, split="val")
    te_ds = ClipDataset(te, cfg, split="test")
    tr_ld = make_train_loader(tr_ds, cfg, class_counts)
    va_ld = DataLoader(va_ds, batch_size=cfg["TRAIN"]["batch_size"], shuffle=False,
                       num_workers=cfg["TRAIN"]["num_workers"],
                       pin_memory=cfg["TRAIN"].get("pin_memory", False))
    te_ld = DataLoader(te_ds, batch_size=cfg["TRAIN"]["batch_size"], shuffle=False,
                       num_workers=cfg["TRAIN"]["num_workers"],
                       pin_memory=cfg["TRAIN"].get("pin_memory", False))

    print("[openfield stage1] head-only ...")
    print(f"   train={len(tr)} val={len(va)} batch={cfg['TRAIN']['batch_size']} "
          f"grad_accum={cfg['TRAIN'].get('grad_accum', 1)} (有效batch="
          f"{cfg['TRAIN']['batch_size'] * cfg['TRAIN'].get('grad_accum', 1)})")
    s1_crit = build_loss(cfg["LOSS"], class_counts)
    if args.resume:
        print("[openfield stage1] 跳过（--resume 续训：head 已存在于 checkpoint）")
    else:
        run_stage(model, "openfield-s1", tr_ld, va_ld, cfg, s1_crit, device, class_names,
                  lr=cfg["TRAIN"]["transfer_stages"]["stage1_lr"],
                  epochs=cfg["TRAIN"]["transfer_stages"]["stage1_epochs"],
                  freeze_first=True, metric_dir=metric_dir,
                  sem_mat=sem_mat, sem_weight=sem_weight, l2sp=l2sp)

    print("[openfield stage2] full fine-tune ...")
    s2_epochs = args.max_epochs or cfg["TRAIN"]["epochs"]
    use_aux = (cfg["TRAIN"].get("aux_homecage", False) and hc_tr_ld is not None
               and hc_crit is not None)
    if use_aux:
        print(f"[aux] 联合训练开启：L = L_openfield + {cfg['TRAIN'].get('aux_weight', 0.3)} * L_homecage")
    print(f"   train={len(tr)} val={len(va)} test={len(te)} epochs={s2_epochs} "
          f"patience={cfg['TRAIN']['early_stop_patience']}")
    best_f1, best_state, best_epoch = run_stage(
        model, "openfield-s2", tr_ld, va_ld, cfg, criterion, device, class_names,
        lr=cfg["TRAIN"]["lr"], epochs=s2_epochs, freeze_first=False,
        metric_dir=metric_dir, patience=cfg["TRAIN"]["early_stop_patience"],
        resume_path=args.resume,
        aux_loader=hc_tr_ld if use_aux else None,
        aux_criterion=hc_crit,
        aux_weight=float(cfg["TRAIN"].get("aux_weight", 0.3)) if use_aux else 0.0,
        sem_mat=sem_mat, sem_weight=sem_weight,
        sem_aux_mat=sem_aux_mat if use_aux else None,
        l2sp=l2sp)

    if best_state is None:
        raise RuntimeError("无可用 best_state（未发生任何优于起始的 epoch）；"
                           "请勿在 --resume 无进展时直接测试，或调低 early_stop_patience")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = os.path.join(metric_dir, "test_metrics.csv")
    test_cm = os.path.join(metric_dir, "confusion_matrices", "test")
    tloss, toverall, tper, tcm = eval_epoch(model, te_ld, criterion, device, class_names,
                                            best_epoch, "test", test_metrics, test_cm)
    print(f"[test] acc={toverall['acc']:.2f} f1_macro={toverall['f1_macro']:.2f} "
          f"f1_weighted={toverall['f1_weighted']:.2f}")
    for k, v in tper.items():
        print(f"    {k}: recall={v['recall']:.2f} P={v['precision']:.2f} F1={v['f1']:.2f}")

    if cfg["POSTPROCESS"].get("threshold_tuning", False):
        from postprocess.thresholds import tune_thresholds, threshold_predict
        print("[postprocess] 验证集逐类阈值调优（DeepEthogram, eLife 2021）...")
        bg_idx = class_names.index("background")
        va_probs, va_labels = collect_probs(model, va_ld, device)
        thresholds = tune_thresholds(
            va_probs, va_labels, bg_idx=bg_idx,
            candidates=int(cfg["POSTPROCESS"].get("threshold_candidates", 40)),
            passes=int(cfg["POSTPROCESS"].get("threshold_passes", 2)),
            min_f1=float(cfg["POSTPROCESS"].get("threshold_min_f1", 0.05)))
        print("   thresholds:", {n: round(float(t), 3) for n, t in zip(class_names, thresholds)})
        te_probs, te_labels = collect_probs(model, te_ld, device)
        te_pred = threshold_predict(te_probs, thresholds, bg_idx=bg_idx)
        th_overall, th_per, th_cm = classwise_and_overall(
            te_labels.tolist(), te_pred.tolist(), class_names)
        print(f"[test][thresholds] acc={th_overall['acc']:.2f} "
              f"f1_macro={th_overall['f1_macro']:.2f} f1_weighted={th_overall['f1_weighted']:.2f}")
        for k, v in th_per.items():
            print(f"    {k}: recall={v['recall']:.2f} P={v['precision']:.2f} F1={v['f1']:.2f}")
        with open(os.path.join(metric_dir, "thresholds.json"), "w") as f:
            json.dump({"thresholds": [float(t) for t in thresholds],
                       "test_f1_macro": th_overall["f1_macro"], "thresholds_seed": cfg["TRAIN"]["seed"]},
                      f, indent=2)
        os.makedirs(os.path.join(metric_dir, "confusion_matrices"), exist_ok=True)
        save_confusion_matrix_csv(
            os.path.join(metric_dir, "confusion_matrices", "test_thresholds_cm.csv"),
            th_cm, class_names, "test_thresholds", best_epoch)

    os.makedirs(cfg["OUTPUT"]["checkpoint_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["OUTPUT"]["checkpoint_dir"], "best_model.pt")
    torch.save({"state_dict": best_state, "class_names": class_names,
                "val_f1_macro": best_f1, "best_epoch": best_epoch,
                "test_overall": toverall, "test_per_class": tper,
                "cfg": cfg, "seed": cfg["TRAIN"]["seed"]}, ckpt)
    print(f"[done] best val f1_macro={best_f1:.2f} (ep{best_epoch}) -> {ckpt}")


def _cosine_lr(epoch, cfg, total_epochs=None):
    """余弦退火（warmup 后），total_epochs 用实际阶段总 epoch（--max_epochs 覆盖后不失真）。"""
    warm = cfg["TRAIN"]["warmup_epochs"]
    epochs = total_epochs or cfg["TRAIN"]["epochs"]
    base = cfg["TRAIN"]["lr"]
    if epoch < warm:
        return base * (epoch + 1) / warm
    t = (epoch - warm) / max(1, epochs - warm)
    return base * 0.5 * (1 + np.cos(np.pi * t))


if __name__ == "__main__":
    main()
