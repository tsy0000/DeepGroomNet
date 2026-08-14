"""
模块B（Solution 2）推理端：序列感知后处理（sequence-informed post-processing）。

在多数投票/中值滤波之上，进一步利用“行为状态转移先验”：
  1) 梳理句法：FaceGroom 通常先于 BodyGroom（Berridge & Fentress 1987；Kalueff 2016）
  2) 行为状态具有时间持续性：短于 min_bout 帧的孤立状态段应被合并到相邻状态

参考：
  - Wilson et al., "Refining ... Behaviour Classifications With Sequence-Informed Post-Processing."
    Ecol Evol 2026. PMID 42544294
  - Berridge & Fentress, Behav Brain Res 1987（梳理句法先验）

实现：
  - 先按连续状态把帧序列分成 bout；
  - 短 bout（< min_bout）合并到前/后相邻 bout 中（按“与前相同或句法前驱”优先）；
  - 可选 transition_prior：FaceGroom -> BodyGroom 视为合法连续，抑制 FaceGroom 被误判为 Locomote。
"""
import numpy as np


def _bout_split(preds):
    """返回 [(start, end, state_id), ...]"""
    preds = np.asarray(preds, dtype=np.int64)
    bouts = []
    n = len(preds)
    i = 0
    while i < n:
        j = i + 1
        while j < n and preds[j] == preds[i]:
            j += 1
        bouts.append([i, j, int(preds[i])])
        i = j
    return bouts


def merge_short_bouts(preds, min_bout=5, syntax_prior=True, class_names=None):
    """
    短 bout 合并：
      - 长度 < min_bout 的段，合并到"前段或后段"中。
      - 若 syntax_prior 且当前段是 FaceGroom，优先并入后段的 BodyGroom（句法前驱关系）。
    class_names: 类 id -> 名称 的映射（必须与模型输出顺序一致，来自 config.openfield_keep）。
    """
    preds = np.asarray(preds, dtype=np.int64).copy()
    bouts = _bout_split(preds)
    if len(bouts) <= 1:
        return preds

    names = {i: n for i, n in enumerate(class_names)} if class_names else {}

    changed = True
    while changed:
        changed = False
        bouts = _bout_split(preds)
        for b in bouts:
            start, end, sid = b
            if end - start >= min_bout:
                continue
            l_idx = None
            r_idx = None
            for k, bb in enumerate(bouts):
                if bb[2] == sid and bb[0] == start:
                    l_idx = k - 1 if k > 0 else None
                    r_idx = k + 1 if k < len(bouts) - 1 else None
                    break
            if l_idx is None and r_idx is None:
                continue
            candidates = []
            if l_idx is not None:
                candidates.append((l_idx, bouts[l_idx][2]))
            if r_idx is not None:
                candidates.append((r_idx, bouts[r_idx][2]))
            pick = candidates[0][0]
            if syntax_prior and names.get(sid) == "FaceGroom":
                for (ci, cs) in candidates:
                    if names.get(cs) == "BodyGroom":
                        pick = ci
                        break
            target_state = bouts[pick][2]
            preds[start:end] = target_state
            changed = True
            break
    return preds


def merge_short_bouts_per_class(preds, min_bout_per_class, class_names=None):
    """DeepEthogram 式逐类短 bout 合并：每类用自己的最小 bout 长度（数据驱动百分位）。

    min_bout_per_class: [C] 每类阈值（bout_length_percentiles 输出）。
    其余语义与 merge_short_bouts 相同（首尾段保留、逐轮扫描直至稳定）。
    """
    preds = np.asarray(preds, dtype=np.int64).copy()
    names = {i: n for i, n in enumerate(class_names)} if class_names else {}
    changed = True
    while changed:
        changed = False
        bouts = _bout_split(preds)
        for k, (start, end, sid) in enumerate(bouts):
            min_b = int(min_bout_per_class[sid]) if sid < len(min_bout_per_class) else 1
            if end - start >= min_b:
                continue
            if k == 0 or k == len(bouts) - 1:
                continue
            l, r = bouts[k - 1], bouts[k + 1]
            pick = (r[2] if (names.get(sid) == "FaceGroom"
                             and names.get(r[2]) == "BodyGroom") else l[2])
            preds[start:end] = pick
            changed = True
            break
    return preds


def _is_groom(sid, names):
    return names.get(sid) in ("FaceGroom", "BodyGroom")


def bout_length_percentiles(label_seqs, class_names=None, percentile=1):
    """DeepEthogram 式数据驱动最小 bout 长度：逐类统计训练标签 bout 长度分布的百分位。

    label_seqs: iterable of 标签序列（np 数组 / list，值为类 id）。
    返回: [C] 每类最小 bout 长度（取整；无该类的类返回 1）。
    参考：DeepEthogram postprocessing.get_bout_length_percentile（每类独立，
    短而稀疏的类不会被统一窗口误杀）。
    """
    from collections import defaultdict
    lens = defaultdict(list)
    for seq in label_seqs:
        s = np.asarray(seq, dtype=np.int64)
        if s.size == 0:
            continue
        idx = np.where(np.diff(np.concatenate(([0], s, [0]))))[0]
        for st, en in zip(idx[::2], idx[1::2]):
            if en - st > 0:
                lens[int(s[st])].append(en - st)
    n = len(class_names) if class_names else (max(lens) + 1 if lens else 1)
    out = np.ones(n, dtype=np.int64)
    for c, ls in lens.items():
        if ls and 0 <= c < n:
            out[c] = max(1, int(np.percentile(ls, percentile)))
    return out


def sakamoto_bout_filter(preds, class_names=None, min_isolated=4, min_gap=6,
                         min_switch=4, min_bout_per_class=None):
    """Sakamoto et al. 2022（Front. Behav. Neurosci. 16:797860）三滤波器（串行反转）。

    原文：3D-CNN 输出的 FG/BG/NG 帧序列含三类"不可能预测"，逐一反转修正：
      (i)   孤立误判：长度 ≤ min_isolated（4帧）的梳理段 → 并入相邻状态；
      (ii)  不合理中断：梳理 bout 内部 ≤ min_gap（6帧）的缺口 → 填充为相邻梳理；
      (iii) 不合理切换：≤ min_switch（4帧）的 FG↔BG 互切段 → 翻转为两侧状态。
    原文效果：测试集 FG 敏感度 81.3%、PPR 83.5%；BG 91.9%/88.5%；
    梳理 bout 数与人工观察高度相关（face r=0.93, body r=0.98）。
    本实现将三类推广到 5 类序列（背景等非梳理类视为"非梳理"）。
    """
    preds = np.asarray(preds, dtype=np.int64).copy()
    names = {i: n for i, n in enumerate(class_names)} if class_names else {}

    bouts = _bout_split(preds)
    for k, (s, e, sid) in enumerate(bouts):
        if e - s > min_switch or k == 0 or k == len(bouts) - 1:
            continue
        if not _is_groom(sid, names):
            continue
        l, r = bouts[k - 1], bouts[k + 1]
        if l[2] == r[2] and _is_groom(l[2], names):
            preds[s:e] = l[2]

    bouts = _bout_split(preds)
    for k, (s, e, sid) in enumerate(bouts):
        if e - s > min_gap or k == 0 or k == len(bouts) - 1:
            continue
        if _is_groom(sid, names):
            continue
        l, r = bouts[k - 1], bouts[k + 1]
        if l[2] == r[2] and _is_groom(l[2], names):
            preds[s:e] = l[2]

    if min_bout_per_class is not None:
        preds = merge_short_bouts_per_class(preds, min_bout_per_class,
                                            class_names=class_names)
    else:
        preds = merge_short_bouts(preds, min_bout=min_isolated, syntax_prior=False,
                                  class_names=class_names)
    return preds


def sequence_decode(preds, cfg, class_names=None, min_bout_per_class=None):
    """模块B 推理端入口。preds: [N] 类 id。class_names 必须与模型输出顺序一致。

    min_bout_per_class: [C] 数据驱动逐类最小 bout 长度（bout_length_percentiles 输出）；
    传入时 (i) 步骤用逐类阈值（DeepEthogram 式），否则用 sakamoto_min_isolated 固定值。
    """
    preds = np.asarray(preds, dtype=np.int64)
    if bool(cfg["POSTPROCESS"].get("sakamoto_filter", True)):
        return sakamoto_bout_filter(
            preds, class_names=class_names,
            min_isolated=int(cfg["POSTPROCESS"].get("sakamoto_min_isolated", 4)),
            min_gap=int(cfg["POSTPROCESS"].get("sakamoto_min_gap", 6)),
            min_switch=int(cfg["POSTPROCESS"].get("sakamoto_min_switch", 4)),
            min_bout_per_class=min_bout_per_class)
    min_bout = int(cfg["POSTPROCESS"].get("seq_min_bout", 5))
    syntax_prior = bool(cfg["POSTPROCESS"].get("transition_prior", True))
    return merge_short_bouts(preds, min_bout=min_bout, syntax_prior=syntax_prior,
                             class_names=class_names)
