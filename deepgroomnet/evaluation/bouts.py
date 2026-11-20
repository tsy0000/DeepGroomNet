import numpy as np


def _runs(mask):
    m = np.asarray(mask, dtype=bool)
    changes = np.flatnonzero(np.diff(m.astype(np.int8)) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(m)]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if m[a]]


def _merge(runs, gap_frames):
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap_frames:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def bout_table(labels, class_names, fps=30.0, merge_gap_frames=5, min_bout_frames=0):
    labels = np.asarray(labels, dtype=int)
    rows = {}
    for index, name in enumerate(class_names):
        runs = _merge(_runs(labels == index), merge_gap_frames)
        durations = np.array([(b - a) / fps for a, b in runs if (b - a) >= min_bout_frames])
        rows[name] = {
            "n_bouts": int(len(durations)),
            "mean_bout_s": float(durations.mean()) if len(durations) else 0.0,
            "median_bout_s": float(np.median(durations)) if len(durations) else 0.0,
            "max_bout_s": float(durations.max()) if len(durations) else 0.0,
            "total_s": float(durations.sum()),
        }
    return rows


def grooming_summary(labels, class_names, fps=30.0, merge_gap_s=6.0,
                     grooming=("FaceGroom", "BodyGroom")):
    labels = np.asarray(labels, dtype=int)
    indices = [class_names.index(g) for g in grooming if g in class_names]
    mask = np.isin(labels, indices)
    runs = _merge(_runs(mask), int(round(merge_gap_s * fps)))
    durations = np.array([(b - a) / fps for a, b in runs])
    total = float(mask.sum() / fps)
    return {
        "total_grooming_s": total,
        "n_bouts": int(len(durations)),
        "time_per_bout_s": float(total / len(durations)) if len(durations) else 0.0,
        "mean_bout_s": float(durations.mean()) if len(durations) else 0.0,
        "face_s": float((labels == class_names.index("FaceGroom")).sum() / fps)
        if "FaceGroom" in class_names else 0.0,
        "body_s": float((labels == class_names.index("BodyGroom")).sum() / fps)
        if "BodyGroom" in class_names else 0.0,
    }


def face_body_switches(labels, class_names, fps=30.0, merge_gap_s=6.0):
    labels = np.asarray(labels, dtype=int)
    face = class_names.index("FaceGroom")
    body = class_names.index("BodyGroom")
    groom = np.isin(labels, [face, body])
    runs = _merge(_runs(groom), int(round(merge_gap_s * fps)))
    forward = backward = 0
    for a, b in runs:
        stages = [int(v) for v in labels[a:b]]
        stages = [s for s in stages if s in (face, body)]
        collapsed = [stages[0]] if stages else []
        for value in stages[1:]:
            if value != collapsed[-1]:
                collapsed.append(value)
        for first, second in zip(collapsed[:-1], collapsed[1:]):
            if first == face and second == body:
                forward += 1
            elif first == body and second == face:
                backward += 1
    return {"face_to_body": forward, "body_to_face": backward,
            "total": forward + backward}
