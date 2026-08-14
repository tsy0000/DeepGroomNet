"""
音节多数投票：在帧级预测之上按 MoSeq 音节边界聚合，保留"自然分界"。

设计依据（用户目标"推理用 MoSeq 保证自然分界"）：
  - 分类器逐帧预测在音节内部可能抖动；
  - MoSeq 音节是数据驱动的自然行为单元，其边界应被尊重（不跨音节投票）；
  - 因此：每个音节段内部做多数投票 -> 该段统一为占优行为状态，
    音节边界保持为行为分段（自然分界），再交给 sequence_decode 做短 bout 合并。

参考：
  - Weinreb et al., "Keypoint-MoSeq: parsing behavior by linking point tracking to
    pose dynamics." (Nat Methods 2024 / bioRxiv 2023.03.16.532307)
  - Wilson et al., "Refining ... Behaviour Classifications With Sequence-Informed
    Post-Processing." Ecol Evol 2026. PMID 42544294
"""
from __future__ import annotations

import numpy as np


def syllable_majority_vote(preds, syllables, min_syllable_len=1):
    """按音节段内多数投票重写预测。

    preds     : np.ndarray [N] 帧级类 id
    syllables : np.ndarray [N] 帧级音节 id（MoSeq/Keypoint-MoSeq 输出）
    min_syllable_len: 短于该长度的音节不强制投票（保留原始帧级预测，避免噪声音节污染）

    返回 np.ndarray [N]（dtype 与 preds 一致）。
    """
    preds = np.asarray(preds, dtype=np.int64).copy()
    syl = np.asarray(syllables)
    if len(preds) != len(syl):
        raise ValueError(f"preds({len(preds)}) 与 syllables({len(syl)}) 长度不一致")
    n = len(preds)
    if n == 0:
        return preds

    i = 0
    while i < n:
        j = i + 1
        while j < n and syl[j] == syl[i]:
            j += 1
        if j - i >= min_syllable_len:
            seg = preds[i:j]
            vals, counts = np.unique(seg, return_counts=True)
            preds[i:j] = vals[int(np.argmax(counts))]
        i = j
    return preds


def apply_syllable_pipeline(preds, syllables, cfg, class_names=None):
    """推理端完整后处理管线（顺序固定）：
      1) 音节多数投票（保留 MoSeq 自然分界）
      2) 短 bout 合并 + 可选句法先验（sequence_decode）
    返回 (preds_after_vote, preds_final)。
    """
    from postprocess.sequence_decode import sequence_decode

    after_vote = syllable_majority_vote(
        preds, syllables,
        min_syllable_len=int(cfg["POSTPROCESS"].get("syllable_min_len", 1)))

    method = cfg["POSTPROCESS"].get("smoothing", "sequence")
    if method == "sequence":
        final = sequence_decode(after_vote, cfg, class_names=class_names)
    else:
        from postprocess.temporal_smoothing import smooth
        final = smooth(after_vote, method=method,
                       window=cfg["POSTPROCESS"].get("vote_window", 5))
    return after_vote, final
