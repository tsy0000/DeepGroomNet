import numpy as np

def _bout_split(preds):
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
    preds = np.asarray(preds, dtype=np.int64).copy()
    bouts = _bout_split(preds)
    if len(bouts) <= 1:
        return preds
    names = {i: n for (i, n) in enumerate(class_names)} if class_names else {}
    changed = True
    while changed:
        changed = False
        bouts = _bout_split(preds)
        for b in bouts:
            (start, end, sid) = b
            if end - start >= min_bout:
                continue
            l_idx = None
            r_idx = None
            for (k, bb) in enumerate(bouts):
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
            if syntax_prior and names.get(sid) == 'FaceGroom':
                for (ci, cs) in candidates:
                    if names.get(cs) == 'BodyGroom':
                        pick = ci
                        break
            target_state = bouts[pick][2]
            preds[start:end] = target_state
            changed = True
            break
    return preds

def _is_groom(sid, names):
    return names.get(sid) in ('FaceGroom', 'BodyGroom')

def sakamoto_bout_filter(preds, class_names=None, min_isolated=4, min_gap=6, min_switch=4):
    preds = np.asarray(preds, dtype=np.int64).copy()
    names = {i: n for (i, n) in enumerate(class_names)} if class_names else {}
    bouts = _bout_split(preds)
    for (k, (s, e, sid)) in enumerate(bouts):
        if e - s > min_switch or k == 0 or k == len(bouts) - 1:
            continue
        if not _is_groom(sid, names):
            continue
        (l, r) = (bouts[k - 1], bouts[k + 1])
        if l[2] == r[2] and _is_groom(l[2], names):
            preds[s:e] = l[2]
    bouts = _bout_split(preds)
    for (k, (s, e, sid)) in enumerate(bouts):
        if e - s > min_gap or k == 0 or k == len(bouts) - 1:
            continue
        if _is_groom(sid, names):
            continue
        (l, r) = (bouts[k - 1], bouts[k + 1])
        if l[2] == r[2] and _is_groom(l[2], names):
            preds[s:e] = l[2]
    preds = merge_short_bouts(preds, min_bout=min_isolated, syntax_prior=False, class_names=class_names)
    return preds

def sequence_decode(preds, cfg, class_names=None):
    preds = np.asarray(preds, dtype=np.int64)
    if bool(cfg['POSTPROCESS'].get('sakamoto_filter', True)):
        return sakamoto_bout_filter(preds, class_names=class_names, min_isolated=int(cfg['POSTPROCESS'].get('sakamoto_min_isolated', 4)), min_gap=int(cfg['POSTPROCESS'].get('sakamoto_min_gap', 6)), min_switch=int(cfg['POSTPROCESS'].get('sakamoto_min_switch', 4)))
    min_bout = int(cfg['POSTPROCESS'].get('seq_min_bout', 5))
    syntax_prior = bool(cfg['POSTPROCESS'].get('transition_prior', True))
    return merge_short_bouts(preds, min_bout=min_bout, syntax_prior=syntax_prior, class_names=class_names)
