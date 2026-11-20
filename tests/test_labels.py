import numpy as np

from deepgroomnet import CLASS_NAMES
from deepgroomnet.postprocessing.labels import probs_to_labels, write_ethogram


def test_class_order():
    assert CLASS_NAMES == ("Locomote", "Rear", "FaceGroom", "BodyGroom", "background")


def test_thresholds_and_background():
    probs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2],
                      [0.1, 0.1, 0.9, 0.2, 0.1],
                      [0.1, 0.1, 0.8, 0.95, 0.1]])
    thresholds = [0.45, 0.55, 0.79, 0.75, 0.63]
    result = probs_to_labels(probs, thresholds, list(CLASS_NAMES))
    assert list(result["labels"]) == [4, 2, 3]


def test_sgatf_path_runs():
    probs = np.full((30, 5), 0.2)
    probs[10:20, 2] = 0.9
    syllables = np.repeat(np.arange(3), 10)
    result = probs_to_labels(probs, [0.45, 0.55, 0.79, 0.75, 0.63],
                             list(CLASS_NAMES), syllables=syllables, use_sgatf=True)
    assert result["alpha"] is not None
    assert len(result["labels"]) == 30


def test_write_ethogram(tmp_path):
    path = tmp_path / "ethogram.csv"
    write_ethogram(np.array([4, 4, 2, 2]), list(CLASS_NAMES), path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert lines[0] == "frame,behavior"
    assert lines[3] == "2,FaceGroom"
