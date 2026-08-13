from __future__ import annotations
import json
import numpy as np

def extract_high_conf_bouts(preds, confs, class_names, target_classes, min_conf=0.85, min_len=8):
    preds = np.asarray(preds, dtype=np.int64)
    confs = np.asarray(confs, dtype=np.float32)
    n = len(preds)
    target_ids = {class_names.index(c) for c in target_classes}
    out = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and preds[j] == preds[i]:
            j += 1
        if preds[i] in target_ids and j - i >= min_len:
            c = confs[i:j].mean()
            if c >= min_conf:
                out.append((int(i), int(j - 1), class_names[int(preds[i])], float(c)))
        i = j
    return sorted(out, key=lambda b: b[3], reverse=True)

def build_validation_prompt(bout, class_names, language='zh'):
    (s, e, cls, conf) = bout
    dur = e - s + 1
    if language == 'zh':
        prompt = f'你是一名动物行为学专家。以下是自动行为识别模型在某只小鼠视频中检测到的一个行为片段，候选作为训练伪标签。\n\n【候选片段】\n- 行为类别: {cls}\n- 帧区间: [{s}, {e}]，时长 {dur} 帧（约 {dur / 30.0:.1f} 秒 @ 30fps）\n- 模型平均置信度: {conf:.3f}\n\n【判断标准】\n请从行为学角度判断这个片段"作为 {cls} 的训练标签是否合理"。考虑：\n1. 时长合理性：{dur} 帧（{dur / 30.0:.1f} 秒）对于 {cls} 是否是一个自然持续时长？（FaceGroom/BodyGroom 通常持续 0.5-15 秒）\n2. 类别合理性：{cls} 的典型运动模式（面部/身体梳理的节奏性动作）是否与"连续 {dur} 帧"一致？\n\n【输出】只输出一个 JSON 对象，不要任何解释：\n{{"valid": true/false, "reason": "一句话理由"}}\n'
    else:
        prompt = f'You are an ethology expert. An automatic behavior classifier detected a candidate bout in a mouse video, proposed as a pseudo training label.\n\n[Candidate bout]\n- Class: {cls}\n- Frame interval: [{s}, {e}], duration {dur} frames (~{dur / 30.0:.1f} s @ 30fps)\n- Mean model confidence: {conf:.3f}\n\n[Criteria]\nIs this bout reasonable as a training label for {cls}?\n1. Duration: is {dur} frames ({dur / 30.0:.1f} s) a natural duration for {cls}? (FaceGroom/BodyGroom typically last 0.5-15 s)\n2. Class: is the typical motion pattern of {cls} (rhythmic face/body grooming) consistent with {dur} consecutive frames?\n\n[Output] Output ONLY a JSON object, no explanation:\n{{"valid": true/false, "reason": "one sentence"}}\n'
    return prompt

def parse_validation(response):
    if not response:
        return (False, 'empty')
    text = response.strip()
    if text.startswith('```'):
        text = text.split('```')[1] if '```' in text[3:] else text
        text = text.strip().lstrip('json').strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return (False, 'parse_failed')
    if not isinstance(d, dict) or 'valid' not in d:
        return (False, 'parse_failed')
    return (bool(d['valid']), str(d.get('reason', '')))

def validate_bout(bout, client, class_names, language='zh'):
    if client is None:
        return (True, 'no_llm_rule_based')
    prompt = build_validation_prompt(bout, class_names, language=language)
    try:
        text = client.complete(prompt, system='只输出 JSON。', temperature=0.0, max_tokens=200)
    except Exception as e:
        return (False, f'llm_error: {e}')
    return parse_validation(text)

def select_pseudo_labels(preds, confs, class_names, target_classes, client=None, min_conf=0.85, min_len=8, max_samples=200, language='zh'):
    cands = extract_high_conf_bouts(preds, confs, class_names, target_classes, min_conf=min_conf, min_len=min_len)
    accepted = []
    for b in cands[:max_samples]:
        (ok, reason) = validate_bout(b, client, class_names, language=language)
        if ok:
            accepted.append(b)
    return accepted
