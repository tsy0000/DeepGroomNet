import numpy as np

from deepgroomnet import CLASS_NAMES
from deepgroomnet.evaluation.bouts import bout_table, face_body_switches, grooming_summary


def build_labels(pattern):
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    return np.array([mapping[p] for p in pattern], dtype=int)


def test_bout_table_basic():
    labels = build_labels(["background"] * 10 + ["FaceGroom"] * 30 + ["background"] * 10)
    table = bout_table(labels, list(CLASS_NAMES), fps=30)
    assert table["FaceGroom"]["n_bouts"] == 1
    assert abs(table["FaceGroom"]["mean_bout_s"] - 1.0) < 1e-9


def test_merge_gap_and_minimum():
    labels = build_labels(["FaceGroom"] * 30 + ["background"] * 30 + ["FaceGroom"] * 30)
    merged = bout_table(labels, list(CLASS_NAMES), fps=30, merge_gap_frames=5)
    assert merged["FaceGroom"]["n_bouts"] == 2
    gap_merged = bout_table(labels, list(CLASS_NAMES), fps=30, merge_gap_frames=31)
    assert gap_merged["FaceGroom"]["n_bouts"] == 1


def test_grooming_summary_six_second_rule():
    labels = build_labels(["FaceGroom"] * 30 + ["background"] * 60 + ["BodyGroom"] * 30)
    split = grooming_summary(labels, list(CLASS_NAMES), fps=30, merge_gap_s=1.0)
    joined = grooming_summary(labels, list(CLASS_NAMES), fps=30, merge_gap_s=3.0)
    assert split["n_bouts"] == 2
    assert joined["n_bouts"] == 1
    assert abs(joined["total_grooming_s"] - 2.0) < 1e-9


def test_face_body_switches():
    labels = build_labels(["FaceGroom"] * 30 + ["BodyGroom"] * 30 +
                          ["FaceGroom"] * 30 + ["BodyGroom"] * 30)
    result = face_body_switches(labels, list(CLASS_NAMES), fps=30, merge_gap_s=6.0)
    assert result["face_to_body"] == 2
    assert result["body_to_face"] == 1
    assert result["total"] == 3
