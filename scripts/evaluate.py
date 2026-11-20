import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet.config import load_config
from deepgroomnet.evaluation.bouts import bout_table, face_body_switches, grooming_summary
from deepgroomnet.evaluation.stats import bh_fdr, cliff_delta, exact_permutation_p


def load_group(manifest):
    frame = pd.read_csv(manifest)
    groups = {}
    for _, row in frame.iterrows():
        labels = np.load(row["labels"]) if str(row["labels"]).endswith(".npy") else \
            np.array([int(v) for v in str(row["labels"]).split()])
        groups.setdefault(row["group"], []).append((row["session"], labels))
    return groups


def main():
    parser = argparse.ArgumentParser(description="Bout metrics and two-group statistics")
    parser.add_argument("--manifest", required=True,
                        help="CSV with session, group, labels (path to .npy or space-separated)")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--out", default="runs/evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fps = float(cfg["EVALUATION"]["fps"])
    gap = float(cfg["EVALUATION"]["bout_merge_gap_s"])
    names = cfg["CLASSES"]["names"]
    groups = load_group(args.manifest)

    rows = []
    for group, items in groups.items():
        for session, labels in items:
            summary = grooming_summary(labels, names, fps=fps, merge_gap_s=gap)
            switches = face_body_switches(labels, names, fps=fps, merge_gap_s=gap)
            bouts = bout_table(labels, names, fps=fps,
                               merge_gap_frames=int(round(gap * fps)))
            rows.append({"group": group, "session": session, **summary, **switches,
                         "groom_bouts_merged": bouts["FaceGroom"]["n_bouts"] +
                                               bouts["BodyGroom"]["n_bouts"]})
    per_animal = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_animal.to_csv(out / "per_animal.csv", index=False)

    metrics = ["total_grooming_s", "n_bouts", "time_per_bout_s", "face_s", "body_s", "total"]
    group_names = sorted(per_animal["group"].unique())
    stats_rows = []
    if len(group_names) == 2:
        a = per_animal[per_animal.group == group_names[0]]
        b = per_animal[per_animal.group == group_names[1]]
        for metric in metrics:
            p = exact_permutation_p(a[metric].to_numpy(), b[metric].to_numpy())
            stats_rows.append({"metric": metric, "group_a": group_names[0],
                               "group_b": group_names[1],
                               "mean_a": float(a[metric].mean()), "mean_b": float(b[metric].mean()),
                               "cliff_delta": float(cliff_delta(a[metric], b[metric])),
                               "p_permutation": float(p)})
        table = pd.DataFrame(stats_rows)
        table["q_bh"] = bh_fdr(table["p_permutation"].to_numpy())
        table.to_csv(out / "group_stats.csv", index=False)
        print(table.round(4).to_string(index=False))
    print(json.dumps({"sessions": int(len(per_animal)), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
