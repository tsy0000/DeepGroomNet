"""
训练过程诊断（train.py 每 N epoch 调用一次）。

目标：训练中直接判断「哪个模块没捕捉到特征 / 哪个模块退化了」，而不用等
最终指标：
  1. 梯度范数（backbone.stages / attentions / head）—— 参数有没有在学
  2. 注意力门统计（各 stage 的 w_s）：
       gate_mean / gate_sat（>0.9 占比）—— 饱和≈1 = 退化为恒等（模块无效）；
       gate 参与比（单点集中~小 / 扩散~大）—— 有没有聚焦"点位"；
       on_mouse - off_mouse 门差 —— 掩码先验是否真的把门压到背景像素
  3. 类 logit 统计（真类均值 logit / sigmoid 概率 / 排他间隔）——
       FG/BG 是否可分离、有没有类被整体压制
  4. 可视化：注意力门热图叠加在原始帧上（每 stage 一张 PNG）——
       肉眼直接看门是不是落在嘴部/面部/躯干等行为学正确区域

所有输出写 outputs/metrics/diagnostics/，零侵入：默认只在 val 后跑
diagnostics_val_batches 批（≈128 clips），关闭诊断可置 TRAIN.diagnostics=false。
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

GRAD_GROUPS = [
    ("downsample", "downsample_layers."),
    ("stages", "stages."),
    ("attention", "attentions."),
    ("head", "head."),
    ("aux_head", "aux_head."),
]


class GradTracker:
    """按模块组累积梯度 L2 范数（平均 over steps），冻结参数自然为 0。"""

    def __init__(self, model):
        self.model = model
        self._sum = {g: 0.0 for g, _ in GRAD_GROUPS}
        self._cnt = {g: 0.0 for g, _ in GRAD_GROUPS}
        self._steps = 0

    def update(self):
        self._steps += 1
        sums = {g: 0.0 for g, _ in GRAD_GROUPS}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if not torch.isfinite(g).all():
                continue
            n2 = float(g.pow(2).sum())
            for group, prefix in GRAD_GROUPS:
                if name.startswith(prefix):
                    sums[group] += n2
                    break
        for g in sums:
            self._sum[g] += sums[g]
            self._cnt[g] += 1.0

    def summary(self):
        return {g: float(np.sqrt(self._sum[g] / max(self._cnt[g], 1e-9)))
                for g, _ in GRAD_GROUPS}


def _set_capture(model, on):
    """开/关所有 attention 模块的特征捕获（_capture=True 时 forward 缓存内部量）。"""
    for m in model.attentions.modules():
        if hasattr(m, "_capture"):
            m._capture = on


def _extract_ws(cache):
    """从各注意力模块的捕获缓存取空间门 w_s（不同实现缓存顺序不同）。"""
    if cache is None:
        return None
    if len(cache) in (3, 4):
        return cache[2]
    if len(cache) in (6, 7):
        return cache[4]
    return None


def _extract_mask(cache):
    """取 forward 实际应用的掩码（诊断 on/off 统计用真实 stage 级掩码）。"""
    if cache is None:
        return None
    if len(cache) in (4, 7) and cache[-1] is not None:
        return cache[-1]
    return None


def _gate_stats(ws, mouse_mask, gw, gh):
    """w_s: [B,1,T,H,W]；mouse_mask: [B,1,1,H0,W0] 输入分辨率掩码。"""
    ws = ws.detach().float()
    g = ws.mean(dim=(1, 2))
    stats = {
        "mean": float(ws.mean()),
        "sat": float((ws > 0.9).float().mean()),
        "conc": float((g.sum((1, 2)) ** 2).sum() /
                      ((g ** 2).sum((1, 2)).sum() + 1e-8) / (gw * gh)),
    }
    if mouse_mask is not None:
        mm = (mouse_mask > 0.5).float().mean(dim=1, keepdim=True)
        g_up = torch.nn.functional.interpolate(
            g.unsqueeze(1), size=(mouse_mask.shape[3], mouse_mask.shape[4]),
            mode="nearest")
        on = (g_up * mm).sum() / (mm.sum() + 1e-6)
        off = (g_up * (1 - mm)).sum() / ((1 - mm).sum() + 1e-6)
        stats["on_mouse"] = float(on)
        stats["off_mouse"] = float(off)
    return stats


@torch.no_grad()
def diagnostic_pass(model, loader, cfg, class_names, out_dir, tag,
                    num_batches=8, make_plots=True):
    """在验证集前 num_batches 批上跑诊断，返回结构化结果并落盘 PNG/JSON。

    tag: 如 "s1_ep05"（用于文件名）。返回 (gate_stats, logit_stats, warn_list)。
    """
    import json
    from tqdm import tqdm
    device = next(model.parameters()).device
    model.eval()
    _set_capture(model, True)

    gate_rows = []
    logit_rows = []
    gate_by_class = {}
    warn = []
    save_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(save_dir, exist_ok=True)
    T = int(cfg["CLIP"]["num_input_frames"])

    all_logits, all_y = [], []
    pbar = tqdm(loader, desc="diag", unit="batch", ncols=90,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}]")
    for step, (x, pose, kp_map, y) in enumerate(pbar):
        if step >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        pose = pose.to(device) if pose is not None else None
        kp_map = kp_map.to(device) if kp_map is not None else None
        logits = model(x, pose=pose, kp_map=kp_map).float()
        all_logits.append(logits)
        all_y.append(y)

        if make_plots and step == 0:
            _save_overlays(model, x, cfg, save_dir, tag, T)
        for i, m in enumerate(model.attentions):
            if not hasattr(m, "_capture") or m._capture_cache is None:
                continue
            ws = _extract_ws(m._capture_cache)
            if ws is None:
                continue
            h, w = ws.shape[3], ws.shape[4]
            mask = _extract_mask(m._capture_cache)
            if mask is None and hasattr(m, "_mouse_mask"):
                mask = m._mouse_mask(x)
            s = _gate_stats(ws, mask, w, h)
            s.update({"stage": i, "attn": type(m).__name__})
            gate_rows.append(s)
            per_sample = ws.mean(dim=(1, 2, 3, 4))
            bc = gate_by_class.setdefault(i, {})
            for b in range(ws.shape[0]):
                bc.setdefault(class_names[int(y[b])], []).append(float(per_sample[b]))
            m._capture_cache = None

    _set_capture(model, False)

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    for c, name in enumerate(class_names):
        sel = y == c
        if sel.sum() == 0:
            continue
        lg = logits[sel, c]
        others = logits[sel].clone()
        others[:, c] = -1e9
        sep = float((lg - others.max(1)[0]).mean())
        logit_rows.append({"class": name, "n": int(sel.sum()),
                           "mean_logit": float(lg.mean()),
                           "prob": float(torch.sigmoid(lg).mean()),
                           "sep": sep})

    attn_frozen = not any(p.requires_grad for p in model.attentions.parameters())
    for s in gate_rows:
        if s["mean"] > 0.97:
            warn.append(f"stage{s['stage']}({s['attn']}) 门均值 {s['mean']:.3f}≈1："
                        f"退化为恒等，模块可能未生效")
        elif 0.45 < s["mean"] < 0.55:
            warn.append(f"stage{s['stage']}({s['attn']}) 门均值 {s['mean']:.3f}≈0.5："
                        f"未分化（sigmoid 饱和前段，检查学习率/初始化）")
        if not attn_frozen and "on_mouse" in s and (s["on_mouse"] - s["off_mouse"]) < 0.02:
            warn.append(f"stage{s['stage']}({s['attn']}) 门在小鼠内/外无差异："
                        f"掩码先验或空间门未聚焦小鼠")
    if not attn_frozen:
        for i, bc in gate_by_class.items():
            if len(bc) < 2:
                continue
            means = {k: sum(v) / len(v) for k, v in bc.items()}
            spread = max(means.values()) - min(means.values())
            if spread < 0.03:
                warn.append(f"stage{i} 门对各行为类别无差异化响应（spread={spread:.3f}）："
                            f"该层注意力没有捕捉到类间差异")
    for r in logit_rows:
        if r["prob"] < 0.1:
            warn.append(f"类 {r['class']} 真类概率均值 {r['prob']:.3f}：整体被压制，"
                        f"模型可能从未输出该类")
    warn = list(dict.fromkeys(warn))

    out_json = os.path.join(save_dir, f"diag_{tag}.json")
    gc = {str(i): {k: round(sum(v) / len(v), 3) for k, v in bc.items()}
          for i, bc in gate_by_class.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"gate": gate_rows, "gate_by_class": gc,
                   "logits": logit_rows, "warnings": warn}, f, indent=2)
    return gate_rows, logit_rows, warn, gc


def _save_overlays(model, x, cfg, save_dir, tag, T):
    """注意力门热图叠加在输入帧上（每 stage 一张 PNG，3 帧采样）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[diag] matplotlib 不可用，跳过热图: {e}")
        return
    n_ch = x.shape[1]
    if n_ch > 3:
        x = x[:1, :3].detach().cpu()
    else:
        x = x[:1].detach().cpu()
    mean_v = torch.tensor(cfg["CLIP"]["mean"], dtype=torch.float32).view(3, 1, 1, 1)
    std_v = torch.tensor(cfg["CLIP"]["std"], dtype=torch.float32).view(3, 1, 1, 1)
    x0 = x * std_v + mean_v
    img = (x0.clamp(0, 1).numpy().transpose(0, 2, 3, 4, 1)[0] * 255).astype(np.uint8)
    frames = sorted(set([0, T // 2, T - 1]))
    for i, m in enumerate(model.attentions):
        if not hasattr(m, "_capture") or m._capture_cache is None:
            continue
        ws = _extract_ws(m._capture_cache)
        if ws is None:
            continue
        g = ws[0, 0].cpu().numpy()
        h, w = g.shape[1], g.shape[2]
        fig, axes = plt.subplots(1, len(frames), figsize=(4.5 * len(frames), 4.5))
        for ax, t in zip(np.atleast_1d(axes), frames):
            fr = img[min(t, len(img) - 1)]
            gmap = cv2.resize(g[min(t, len(g) - 1)], (fr.shape[1], fr.shape[0]))
            ax.imshow(fr)
            ax.imshow(gmap, alpha=0.55, cmap="jet", vmin=0.0, vmax=1.0)
            ax.set_title(f"t={min(t, len(img) - 1)}")
            ax.axis("off")
        fig.suptitle(f"stage{i} {type(m).__name__} gate overlay (ep {tag})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"attn_ep{tag}_stage{i}.png"), dpi=110)
        plt.close(fig)
    print(f"[diag] 注意力门热图 -> {save_dir}/attn_ep{tag}_stage*.png")


def format_line(gate_rows, grad_summary):
    """把诊断结果压成一行摘要（epoch 打印用）。"""
    parts = []
    if grad_summary:
        g = grad_summary
        parts.append(f"grad ds={g['downsample']:.2e} st={g['stages']:.2e} "
                     f"att={g['attention']:.2e} hd={g['head']:.2e}")
    if gate_rows:
        rows = {s["stage"]: s for s in gate_rows}
        for i in sorted(rows):
            s = rows[i]
            parts.append(f"st{i}gate={s['mean']:.2f}(sat{s['sat']:.2f})")
    return " | ".join(parts)
