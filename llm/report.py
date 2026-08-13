from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

def load_predictions(pred_csv):
    df = pd.read_csv(pred_csv)
    return (df['frame'].to_numpy(), df['pred_class'].to_numpy(), df['confidence'].to_numpy())

def bouts_from_predictions(pred_classes, confidences=None, min_len=1):
    lab = np.asarray(pred_classes)
    conf = np.asarray(confidences) if confidences is not None else np.full(len(lab), np.nan)
    bouts = []
    n = len(lab)
    i = 0
    while i < n:
        j = i + 1
        while j < n and lab[j] == lab[i]:
            j += 1
        if j - i >= min_len:
            bouts.append((int(i), int(j - 1), str(lab[i]), float(np.nanmean(conf[i:j])) if not np.isnan(conf).all() else None))
        i = j
    return bouts

def behavior_summary(pred_classes, class_names, fps=30):
    names = {n: i for (i, n) in enumerate(class_names)}
    total = len(pred_classes)
    summary = []
    for c in class_names:
        mask = pred_classes == c
        n_frames = int(mask.sum())
        if n_frames == 0:
            summary.append({'behavior': c, 'total_frames': 0, 'duration_s': 0.0, 'percent': 0.0, 'num_bouts': 0})
            continue
        seg = mask.astype(int)
        n_bouts = int(np.sum(np.diff(seg) == 1)) + (1 if seg[0] == 1 else 0)
        summary.append({'behavior': c, 'total_frames': n_frames, 'duration_s': round(n_frames / fps, 2), 'percent': round(n_frames / total * 100, 2), 'num_bouts': n_bouts})
    return summary

def build_prompt(meta, summary, bouts, top_bouts=60, language='zh'):
    n_bouts = len(bouts)
    if n_bouts > top_bouts:
        bouts = sorted(bouts, key=lambda b: b[1] - b[0], reverse=True)[:top_bouts]
    bout_lines = []
    for (s, e, lab, conf) in bouts:
        conf_str = f', conf={conf:.2f}' if conf is not None else ''
        bout_lines.append(f"  [{s}-{e}] {lab} ({e - s + 1}帧, {round((e - s + 1) / meta['fps'], 2)}s{conf_str})")
    stat_lines = []
    for r in summary:
        stat_lines.append(f"  {r['behavior']}: 总时长 {r['duration_s']}s ({r['percent']}%)，共 {r['num_bouts']} 段")
    if language == 'zh':
        prompt = f"""你是一名动物行为学专家。以下是小鼠行为视频的自动识别结果（帧级预测，{meta['fps']}fps，共 {meta['total_frames']} 帧 = {round(meta['total_frames'] / meta['fps'], 1)} 秒）。\n\n【行为类别】{meta['class_names']}\n\n【总体统计】\n{chr(10).join(stat_lines)}\n\n【行为片段序列】（片段 = 连续同行为的帧段；按时间顺序的前 {min(n_bouts, top_bouts)} 个主要片段）\n{(chr(10).join(bout_lines) if bout_lines else '  （无行为片段）')}\n\n请基于以上数据写一份客观、严谨的行为分析报告（500字以内），必须包含：\n1. 各行为占比与梳理行为的出现次数、平均时长、时间分布（如早/中/晚期）\n2. FaceGroom 与 BodyGroom 的顺序关系（是否遵循"先面部后身体"的梳理句法）\n3. 行为转换模式：常见转移（如 Locomote->Rear、FaceGroom->BodyGroom）\n4. 观察到的异常或值得注意的模式（如果存在）\n只依据我提供的数据，不要编造未提供的信息；不确定的地方明确说明。"""
    else:
        prompt = f"You are an ethology expert. Below are automatic frame-level predictions of a mouse behavior video ({meta['fps']} fps, {meta['total_frames']} frames = {round(meta['total_frames'] / meta['fps'], 1)} s).\n\n[Behavior classes] {meta['class_names']}\n\n[Overall statistics]\n{chr(10).join(stat_lines)}\n\n[Bout sequence] (top {min(n_bouts, top_bouts)} bouts by duration)\n{(chr(10).join(bout_lines) if bout_lines else '  (no bouts)')}\n\nWrite a concise, rigorous behavioral analysis report (max 500 words): behavior proportions, grooming bout count/mean duration/timing, FaceGroom->BodyGroom syntax, transition patterns, and any notable anomalies. Use only the provided data; state explicitly when information is missing."
    return prompt

def generate_report(cfg, pred_csv, syllable_csv=None, out_md=None, language=None):
    from llm.client import build_llm_client, LLMClientError
    llm_cfg = cfg.get('LLM', {})
    language = language or llm_cfg.get('language', 'zh')
    fps = int(cfg['CLIP']['fps'])
    class_names = list(cfg['CLASSES']['openfield_keep'])
    (frames, pred_classes, confs) = load_predictions(pred_csv)
    meta = {'fps': fps, 'total_frames': int(len(frames)), 'class_names': class_names}
    summary = behavior_summary(pred_classes, class_names, fps)
    bouts = bouts_from_predictions(pred_classes, confs)
    prompt = build_prompt(meta, summary, bouts, language=language)
    client = build_llm_client(cfg)
    if client is None:
        print('[llm] provider=none，跳过 LLM，仅输出统计数据')
        os.makedirs(os.path.dirname(out_md) if out_md else '.', exist_ok=True)
        if out_md:
            with open(out_md, 'w', encoding='utf-8') as f:
                f.write('# 行为统计（未调用 LLM）\n\n')
                for r in summary:
                    f.write(f"- {r['behavior']}: {r['duration_s']}s ({r['percent']}%), {r['num_bouts']}段\n")
        return None
    system = '你是一名严谨的动物行为学专家，只依据给定数据做客观分析，不编造信息，不用不确定的语气下结论。' if language == 'zh' else 'You are a rigorous ethology expert. Analyze only the given data; do not fabricate information.'
    try:
        text = client.complete(prompt, system=system, temperature=llm_cfg.get('temperature', 0.3), max_tokens=int(llm_cfg.get('max_tokens', 1500)))
    except LLMClientError as e:
        print(f'[llm] 调用失败（保留统计输出）: {e}')
        text = None
    if out_md:
        os.makedirs(os.path.dirname(out_md) or '.', exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('# 行为分析报告\n\n')
            f.write(f'- 输入: {os.path.basename(pred_csv)}\n')
            f.write(f"- 时长: {meta['total_frames'] / meta['fps']:.1f}s @ {meta['fps']}fps\n\n")
            f.write('## 统计\n\n')
            for r in summary:
                f.write(f"- {r['behavior']}: {r['duration_s']}s ({r['percent']}%), {r['num_bouts']}段\n")
            f.write('\n## LLM 分析\n\n')
            f.write(text if text else '（LLM 调用失败，未生成分析）\n')
    return text
