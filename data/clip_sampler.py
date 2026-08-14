"""
Clip 采样与切分（模块B 源头 + 推理端对齐）。

三条规范，保证 train/test 一致、可发表：
  1) 帧索引约定：视频第 i 帧（0 起始）对 should label 第 i 行（帧级 one-hot / syllable）。
  2) clip 帧采样：从 [start, end) 区间内**均匀抽取** num_input_frames 帧（np.linspace 取整），
     覆盖整个区间（不丢首尾、不重复）；推理端与训练端用同一函数，消除"训练人工段/测试音节段"差异。
  3) 训练增强：boundary_jitter —— 起点在标注边界附近随机抖动（可越过边界），
     让模型对"分段位置不理想"鲁棒（对应模块B源头）。

实现：
  sample_frames            : 区间内均匀抽帧
  segment_to_windows       : 统一窗口生成器（训练段/推理音节段共用，保证窗口结构完全一致）
  extract_uniform_clips    : 训练：段内滑窗 + 段内标签全同（步进 = clip_interval，与推理滑窗一致）
  extract_sliding_clips    : 训练：随机滑窗（可跨边界、中心帧标签），seed 化保证可复现
  homecage_clips           : 预裁剪 mpg/mp4 整段切窗（迁移中间域）
  infer_sliding_windows    : 推理：帧级滑窗（步进 = clip_interval，与训练 uniform 同结构）
  infer_syllable_clips     : （保留）按音节边界切窗——仅在"音节整窗预测"模式下使用
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

OPENFIELD_CLASSES = ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"]


def _cv2():
    """惰性导入 cv2（纯逻辑函数不依赖它，便于无 OpenCV 环境测试/复用）。"""
    import cv2
    return cv2


def sample_frames(start: int, end: int, n_frames: int) -> List[int]:
    """在 [start, end) 内均匀抽取 n_frames 帧（覆盖首尾，不重复）。"""
    n = end - start
    if n <= 0:
        return []
    if n_frames <= 0:
        return []
    if n == 1:
        return [start]
    idx = np.linspace(start, end - 1, num=n_frames).round().astype(int)
    return [int(i) for i in np.unique(idx)]


def read_frames(video_path: str, frame_indices: List[int], fallback_size=(480, 480, 3)):
    """按绝对帧号读取帧（cv2 seek），失败返回占位灰帧。"""
    cv2 = _cv2()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in frame_indices:
        if i >= total or i < 0:
            frames.append(np.zeros(fallback_size, np.uint8))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        frames.append(fr if ok else np.zeros(fallback_size, np.uint8))
    cap.release()
    return frames


def video_frame_count(video_path: str) -> int:
    cv2 = _cv2()
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def normalize_label_columns(labels_df):
    """把标注变体统一为规范列名，返回副本。

    实测两数据集列名差异（2026-08 数据审计）：
      - 'Background'/'background'（大小写变体，datasetssap3 混用）
      - 'Jump'（datasetssap3 额外类：跳跃，归 background——非理毛、无独立行为意义）
    统一后 label_of_window 才能正确匹配 OPENFIELD_CLASSES。
    """
    df = labels_df.copy()
    rename = {}
    for c in df.columns:
        if c in ("Background",) and c != "background":
            rename[c] = "background"
        if c in ("Jump", "Defecate", "defecate") and c != "background":
            rename[c] = "background"
    if rename:
        df = df.rename(columns=rename)
    return df


def label_of_window(labels_df, start: int, end: int) -> str | None:
    """若 [start,end) 内所有帧标签一致且属于 OPENFIELD_CLASSES 则返回该标签，否则 None。"""
    sl = labels_df.iloc[start:end]
    for c in OPENFIELD_CLASSES:
        if c in sl.columns and int(sl[c].sum()) == len(sl):
            return c
    return None


def segment_to_windows(segments: List[Tuple[int, int]], seg: int,
                       step: int = None) -> List[Tuple[int, int]]:
    """把段边界列表切成固定长度窗口（训练标注段 / 推理音节段共用）。

    与推理端完全相同的结构：
      - 段内按 step 步进切窗（step 默认 = seg，即不重叠）
      - 末段不足 seg 时"末对齐"补一个窗口（不丢帧，也不跨段）
    保证训练窗口与推理窗口的（长度, 覆盖, 步进）分布一致。

    segments: [(start, end), ...]（已排序、不重叠）
    """
    if step is None:
        step = seg
    windows = []
    for (i, j) in segments:
        s = i
        while s + seg <= j:
            windows.append((s, s + seg))
            s += step
        if s < j:
            lo = max(i, j - seg)
            windows.append((lo, j))
    return windows


def extract_uniform_clips(video_path, labels_df, cfg, class_names=OPENFIELD_CLASSES):
    """滑窗（间隔 clip_interval），窗口内标签全同才保留。返回 [(start,end,label)]。

    确定性：相同输入产出完全相同的结果（无随机），保证 val/test 划分可复现。
    跨边界抖动是训练期增强（见 dataset.py），不在此处引入随机。

    注意：clip_interval < seg 时窗口重叠——这正是推理端帧级滑窗的结构
    （infer_sliding_windows 使用同一 interval），训练/推理窗口密度一致。
    """
    total = len(labels_df)
    seg = cfg["CLIP"]["orig_segment_len"]
    interval = cfg["CLIP"]["clip_interval"]
    clips = []
    for s in range(0, total - seg + 1, interval):
        e = s + seg
        lab = label_of_window(labels_df, s, e)
        if lab is not None:
            clips.append((s, e, lab))
    return clips


def extract_sliding_clips(video_path, labels_df, cfg, count, class_names=OPENFIELD_CLASSES,
                          seed=None):
    """训练/推理一致化的核心（模块B-源头，方案：滑窗对齐 MoSeq 音节窗口分布）。

    推理时 MoSeq 音节窗口 = "可能含行为边界的 45 帧窗口，标签为主导行为"；
    因此训练也必须见到"含边界的窗口"，否则分布不一致。
    本函数生成 count 个随机滑窗：窗口内标签不要求全同，标签取窗口中心帧的类别
    （与 DeepEthogram 等帧级/滑窗方法的"窗口标签"约定一致，且复现 MoSeq 窗口语义）。

    返回 [(start, end, label)]，label = 中心帧 argmax（仅限 class_names 内）。
    seed 必须传入（config.TRAIN.seed），否则输出不可复现（发表级要求）。
    """
    total = len(labels_df)
    seg = cfg["CLIP"]["orig_segment_len"]
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(count):
        s = int(rng.integers(0, max(1, total - seg)))
        e = s + seg
        center = s + seg // 2
        row = labels_df.iloc[center]
        vals = [float(row[c]) if c in row.index else 0.0 for c in class_names]
        if abs(sum(vals) - 1.0) > 1e-6:
            continue
        clips.append((s, e, class_names[int(np.argmax(vals))]))
    return clips


def jitter_window(start: int, end: int, total: int, jit: int, rng) -> tuple:
    """训练期边界抖动：窗口起点在 ±jit 帧内随机平移（夹紧到视频边界），长度不变。

    平移后窗口仍以原标注段为主（最多 jit 帧越过边界），标签保持不变。
    与 clip 生成完全解耦：val/test 不走此函数，保证评估无随机。
    """
    seg = end - start
    if jit <= 0 or seg <= 0:
        return start, end
    lo = max(0, start - jit)
    hi = min(total - seg, start + jit)
    if hi < lo:
        return start, end
    s = int(rng.integers(lo, hi + 1))
    return s, s + seg


def homecage_clips(clip_path: str, cfg, class_names):
    """把预裁剪的 homecage mpg/mp4 整段切窗（窗口=orig_segment_len，末段不足则用末对齐）。"""
    n = video_frame_count(clip_path)
    seg = cfg["CLIP"]["orig_segment_len"]
    clips = []
    s = 0
    while s + seg <= n:
        clips.append((s, s + seg))
        s += seg
    if n - s > 0 and n - s < seg:
        clips.append((max(0, n - seg), n))
    return clips


def infer_sliding_windows(total_frames: int, cfg) -> List[Tuple[int, int]]:
    """推理：帧级滑窗（与训练 uniform 窗口结构完全一致）。

    步进 = config.CLIP.clip_interval（训练生成 uniform clips 用的同一间隔），
    窗口长度 = orig_segment_len，末段不足时末对齐补一个窗口（不丢帧）。
    每帧被 ceil(seg/interval) 个窗口覆盖 -> 多窗口 logits 平均 = 天然的帧级时间平滑
    （对齐 DeepEthogram/B-SOiD 的帧级滑窗预测范式）。

    注意：训练端 uniform clips 只保留"段内标签全同"的窗口，而推理端所有滑窗都要预测；
    这正是 sliding clips（中心帧标签、可跨边界）进入训练的原因——见 scripts/02_gen_clips.py。
    """
    seg = int(cfg["CLIP"]["orig_segment_len"])
    interval = int(cfg["CLIP"].get("clip_interval", 15))
    if interval <= 0:
        interval = seg
    wins = []
    s = 0
    while s + seg <= total_frames:
        wins.append((s, s + seg))
        s += interval
    if s < total_frames:
        wins.append((max(0, total_frames - seg), total_frames))
    return wins


def infer_syllable_clips(syllable_labels: np.ndarray, cfg) -> List[Tuple[int, int]]:
    """
    按 MoSeq/Keypoint-MoSeq 音节边界切窗（legacy：整窗一个预测模式）。
    syllable_labels: 帧级音节 id（长度=帧数）。
    每个音节段用与训练一致的窗口大小切分，首尾对齐；返回 [(start,end), ...]。

    注意：默认推理流程（infer.py）已改用 infer_sliding_windows 帧级滑窗，
    音节边界只在后处理（syllable_majority_vote）中用于"保留自然分界"，
    不再作为推理窗口本身——除非 config.INFER.window_mode == "syllable"。
    """
    seg = cfg["CLIP"]["orig_segment_len"]
    syl = np.asarray(syllable_labels)
    n = len(syl)
    clips = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and syl[j] == syl[i]:
            j += 1
        s = i
        while s + seg <= j:
            clips.append((s, s + seg))
            s += seg
        if s < j:
            lo = max(i, j - seg)
            clips.append((lo, j))
        i = j
    return clips
