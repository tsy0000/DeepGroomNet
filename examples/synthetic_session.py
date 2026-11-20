import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet import CLASS_NAMES
from deepgroomnet.evaluation.bouts import face_body_switches, grooming_summary
from deepgroomnet.fusion.sgatf import sgatf
from deepgroomnet.postprocessing.labels import write_ethogram


def main():
    rng = np.random.default_rng(0)
    n_frames = 900
    probs = rng.uniform(0.05, 0.95, size=(n_frames, len(CLASS_NAMES)))
    probs[200:320, 2] = 0.9
    probs[200:320, 4] = 0.05
    probs[420:560, 3] = 0.92
    probs[420:560, 4] = 0.04
    syllables = np.repeat(np.arange(30), n_frames // 30)
    thresholds = [0.45, 0.55, 0.79, 0.75, 0.63]

    labels, fused, alpha = sgatf(probs, syllables, thresholds)
    out = Path("runs/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    write_ethogram(labels, list(CLASS_NAMES), out / "ethogram.csv")
    print("alpha range", float(alpha.min()), float(alpha.max()))
    print("labels", {name: int((labels == i).sum()) for i, name in enumerate(CLASS_NAMES)})
    print("grooming", grooming_summary(labels, list(CLASS_NAMES)))
    print("switches", face_body_switches(labels, list(CLASS_NAMES)))
    print("wrote", out / "ethogram.csv")


if __name__ == "__main__":
    main()
