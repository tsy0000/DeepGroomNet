import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet.config import load_config
from deepgroomnet.inference import Segmenter
from deepgroomnet.postprocessing.labels import write_ethogram
from deepgroomnet.preprocessing.detector import load_boxes


def main():
    parser = argparse.ArgumentParser(description="Segment a top-view open-field video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--out", default="runs/infer")
    parser.add_argument("--syllables", help="per-frame syllable .npy from fit_kpms.py/apply_kpms")
    parser.add_argument("--boxes", help="YOLO boxes json keyed by video stem")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["INFERENCE"]["device"]
    boxes = load_boxes(args.boxes) if args.boxes else None
    key = Path(args.video).stem
    segmenter = Segmenter.from_checkpoint(args.checkpoint, cfg, device=device,
                                          boxes=boxes, box_key=key)
    syllables = np.load(args.syllables) if args.syllables else None
    result = segmenter.run(args.video, syllables=syllables)

    out = Path(args.out)
    segmenter.write_outputs(result, out, prefix=key, fps=args.fps)
    summary = {
        "video": str(Path(args.video).resolve()),
        "frames": int(len(result["labels"])),
        "classes": result["class_names"],
        "composition": {name: float((result["labels"] == i).mean())
                        for i, name in enumerate(result["class_names"])},
        "sgatf": syllables is not None,
        "mean_confidence": float(result["probs"].max(axis=1).mean()),
    }
    (out / f"{key}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_ethogram(result["labels"], result["class_names"], out / f"{key}_ethogram.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
