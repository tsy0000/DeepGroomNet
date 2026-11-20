import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepgroomnet.config import load_config
from deepgroomnet.evaluation.bouts import bout_table, grooming_summary
from deepgroomnet.evaluation.stats import bh_fdr, paired_test, bonferroni


def figure_2f(manifest, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps = float(cfg["EVALUATION"]["fps"])
    names = cfg["CLASSES"]["names"]
    frame = pd.read_csv(manifest)
    rows = []
    for _, row in frame.iterrows():
        labels = np.load(row["labels"])
        table = bout_table(labels, names, fps=fps,
                           merge_gap_frames=int(round(float(cfg["EVALUATION"]["bout_merge_gap_s"]) * fps)))
        for behaviour, values in table.items():
            rows.append({"session": row["session"], "day": row["day"],
                         "behaviour": behaviour, **values})
    data = pd.DataFrame(rows)
    durations = data.loc[data.behaviour != "background", "mean_bout_s"]
    q1 = float(np.percentile(durations, 25))
    q3 = float(np.percentile(durations, 75))
    data["short_ratio"] = (data.mean_bout_s < q1).astype(float)
    data["long_ratio"] = (data.mean_bout_s >= q3).astype(float)
    data.to_csv(out / "figure2f_bouts.csv", index=False)
    pairs = [("Day 1", "Day 2"), ("Day 2", "Day 5"), ("Day 1", "Day 5")]
    tests = []
    for metric in ("mean_bout_s", "short_ratio", "long_ratio"):
        for behaviour in names:
            subset = data[data.behaviour == behaviour]
            pivot = subset.pivot(index="session", columns="day", values=metric)
            for first, second in pairs:
                if first in pivot and second in pivot:
                    result = paired_test(pivot[first].dropna(), pivot[second].dropna())
                    tests.append({"metric": metric, "behaviour": behaviour,
                                  "pair": f"{first} vs {second}", **result})
    table = pd.DataFrame(tests)
    table["p_bonferroni"] = bonferroni(table["p"].to_numpy(), 3)
    table.to_csv(out / "figure2f_tests.csv", index=False)
    print(table.round(4).to_string(index=False))


def figure_2g(manifest, cfg, out):
    fps = float(cfg["EVALUATION"]["fps"])
    names = cfg["CLASSES"]["names"]
    frame = pd.read_csv(manifest)
    rows = []
    for _, row in frame.iterrows():
        labels = np.load(row["labels"])
        summary = grooming_summary(labels, names, fps=fps,
                                   merge_gap_s=float(cfg["EVALUATION"]["bout_merge_gap_s"]))
        rows.append({"session": row["session"], "group": row["group"], **summary})
    data = pd.DataFrame(rows)
    data.to_csv(out / "figure2g_per_animal.csv", index=False)
    print(data.groupby("group").mean(numeric_only=True).round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description="Recompute Figure 2F/2G from label files")
    parser.add_argument("--figure", choices=["2f", "2g"], required=True)
    parser.add_argument("--manifest", required=True,
                        help="CSV with session, group, day, labels (path to .npy)")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--out", default="runs/figures")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.figure == "2f":
        figure_2f(args.manifest, cfg, out)
    else:
        figure_2g(args.manifest, cfg, out)
    print(json.dumps({"output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
