from __future__ import annotations
import json
import numpy as np

def summarize_confusion(y_true, y_pred, class_names):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=np.int64)
    for (t, p) in zip(y_true, y_pred):
        cm[t, p] += 1
    pairs = []
    for t in range(k):
        total = int(cm[t].sum())
        if total == 0:
            continue
        for p in range(k):
            if t == p:
                continue
            n = int(cm[t, p])
            if n > 0:
                pairs.append((class_names[t], class_names[p], n, n / total))
    pairs.sort(key=lambda x: (-x[2], x[0]))
    return (cm, pairs)

def sample_error_clips(y_true, y_pred, frame_idx, class_names, pair, max_n=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    (t, p) = pair
    (tid, pid) = (class_names.index(t), class_names.index(p))
    mask = (y_true == tid) & (y_pred == pid)
    idx = np.nonzero(mask)[0]
    bouts = []
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and idx[j] == idx[j - 1] + 1:
            j += 1
        (s, e) = (idx[i], idx[j - 1])
        bouts.append((int(s), int(e), int(e - s + 1)))
        i = j
    out = []
    for (s, e, n) in bouts[:max_n]:
        gs = int(frame_idx[s]) if frame_idx is not None else s
        ge = int(frame_idx[e]) if frame_idx is not None else e
        out.append((gs, ge, n))
    return out

def build_analysis_prompt(cm, pairs, samples, class_names, language='zh'):
    k = len(class_names)
    cm_lines = [' | '.join(['true\\pred'] + list(class_names))]
    for t in range(k):
        cm_lines.append(' | '.join([class_names[t]] + [str(int(cm[t, p])) for p in range(k)]))
    pair_lines = []
    for (t, p, n, frac) in pairs[:8]:
        pair_lines.append(f'{t}->{p}: {n} 帧 ({frac * 100:.1f}% of {t})')
    sample_lines = []
    for (t, p, n, _) in pairs[:4]:
        ss = samples.get((t, p), [])
        for (vid, gs, ge, ln) in ss[:3]:
            src = vid if vid else '本视频'
            sample_lines.append(f'{t}(真)->{p}(预测): {src} 帧[{gs}-{ge}] 长{ln}帧')
    if language == 'zh':
        return f"""你是一名动物行为学专家。以下是小鼠行为识别模型在测试集上的错误分析（第 5 类"background"为非行为背景帧）。\n\n【混淆矩阵】（行=真实类，列=预测类）\n{chr(10).join(cm_lines)}\n\n【主要错误对】（按错误帧数降序）\n{(chr(10).join(pair_lines) if pair_lines else '(无错误对)')}\n\n【错误证据片段】（帧号是视频内全局帧号）\n{(chr(10).join(sample_lines) if sample_lines else '(无片段)')}\n\n【任务】从行为学角度分析错误模式，输出 JSON（不要任何解释）：\n{{\n  "error_modes": [\n    {{\n      "mode": "简要错误模式名",\n      "true_class": "真实类",\n      "pred_class": "被误判为",\n      "likely_cause": "行为学上可能的原因（行为相似性/标注歧义/背景误检等）",\n      "severity": "high|medium|low",\n      "fixable": true/false,\n      "suggestion": "若能修复，具体建议（数据/后处理/模型）"\n    }}\n  ],\n  "summary": "3-5 句总体总结"\n}}"""
    return f"""You are an ethology expert. Below is the error analysis of a mouse behavior recognition model on the test set (class 5 "background" = non-behavior frames).\n\n[Confusion matrix] (rows=true, cols=pred)\n{chr(10).join(cm_lines)}\n\n[Top error pairs] (by frame count)\n{(chr(10).join(pair_lines) if pair_lines else '(none)')}\n\n[Evidence clips] (frame indices are global video frames)\n{(chr(10).join(sample_lines) if sample_lines else '(none)')}\n\n[Task] Analyze error modes from an ethological perspective. Output JSON only, no explanation:\n{{\n  "error_modes": [\n    {{\n      "mode": "short error mode name",\n      "true_class": "true class",\n      "pred_class": "misclassified as",\n      "likely_cause": "ethological cause (behavioral similarity / label ambiguity / background false positive etc.)",\n      "severity": "high|medium|low",\n      "fixable": true/false,\n      "suggestion": "concrete suggestion (data/post-processing/model) if fixable"\n    }}\n  ],\n  "summary": "3-5 sentence overall summary"\n}}"""

def parse_analysis(response):
    if not response:
        return (None, 'empty')
    text = response.strip()
    if text.startswith('```'):
        text = text.split('```')[1] if '```' in text[3:] else text
        text = text.strip().lstrip('json').strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return (None, 'parse_failed')
    if not isinstance(d, dict):
        return (None, 'parse_failed')
    return (d, 'ok')

def run_error_analysis(y_true, y_pred, frame_idx, class_names, client, language='zh'):
    (cm, pairs) = summarize_confusion(y_true, y_pred, class_names)
    report = {'confusion_matrix': cm.tolist(), 'top_error_pairs': [{'true': t, 'pred': p, 'frames': n, 'frac': round(f, 4)} for (t, p, n, f) in pairs[:8]]}
    if client is None:
        report['llm'] = None
        report['llm_status'] = 'skipped'
        return (report, cm)
    samples = {}
    for (t, p, n, _) in pairs[:4]:
        ss = sample_error_clips(y_true, y_pred, frame_idx, class_names, (t, p))
        samples[t, p] = [(None, gs, ge, ln) for (gs, ge, ln) in ss]
    return run_analysis_prompt(report, cm, pairs, samples, class_names, client, language)

def run_error_analysis_videos(videos, class_names, client, language='zh'):
    (ys, ps) = ([], [])
    for (_name, yt, yp) in videos:
        ys.append(np.asarray(yt, dtype=np.int64))
        ps.append(np.asarray(yp, dtype=np.int64))
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)
    (cm, pairs) = summarize_confusion(y_true, y_pred, class_names)
    report = {'confusion_matrix': cm.tolist(), 'top_error_pairs': [{'true': t, 'pred': p, 'frames': n, 'frac': round(f, 4)} for (t, p, n, f) in pairs[:8]]}
    if client is None:
        report['llm'] = None
        report['llm_status'] = 'skipped'
        return (report, cm)
    samples = {}
    for (t, p, n, _) in pairs[:4]:
        entries = []
        for (name, yt, yp) in videos:
            (yt_a, yp_a) = (np.asarray(yt, dtype=np.int64), np.asarray(yp, dtype=np.int64))
            frame_idx = np.arange(len(yt_a))
            for (gs, ge, ln) in sample_error_clips(yt_a, yp_a, frame_idx, class_names, (t, p), max_n=2):
                entries.append((name, gs, ge, ln))
        samples[t, p] = entries[:4]
    return run_analysis_prompt(report, cm, pairs, samples, class_names, client, language)

def run_analysis_prompt(report, cm, pairs, samples, class_names, client, language):
    prompt = build_analysis_prompt(cm, pairs, samples, class_names, language=language)
    try:
        text = client.complete(prompt, system='只输出 JSON。', temperature=0.0, max_tokens=2000)
    except Exception as e:
        report['llm'] = None
        report['llm_status'] = f'llm_error: {e}'
        return (report, cm)
    (d, status) = parse_analysis(text)
    report['llm'] = d
    report['llm_status'] = status
    return (report, cm)
