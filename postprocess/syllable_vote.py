from __future__ import annotations
import numpy as np

def syllable_majority_vote(preds, syllables, min_syllable_len=1):
    preds = np.asarray(preds, dtype=np.int64).copy()
    syl = np.asarray(syllables)
    if len(preds) != len(syl):
        raise ValueError(f'preds({len(preds)}) 与 syllables({len(syl)}) 长度不一致')
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
            (vals, counts) = np.unique(seg, return_counts=True)
            preds[i:j] = vals[int(np.argmax(counts))]
        i = j
    return preds

def apply_syllable_pipeline(preds, syllables, cfg, class_names=None):
    from postprocess.sequence_decode import sequence_decode
    after_vote = syllable_majority_vote(preds, syllables, min_syllable_len=int(cfg['POSTPROCESS'].get('syllable_min_len', 1)))
    method = cfg['POSTPROCESS'].get('smoothing', 'sequence')
    if method == 'sequence':
        final = sequence_decode(after_vote, cfg, class_names=class_names)
    else:
        from postprocess.temporal_smoothing import smooth
        final = smooth(after_vote, method=method, window=cfg['POSTPROCESS'].get('vote_window', 5))
    return (after_vote, final)
