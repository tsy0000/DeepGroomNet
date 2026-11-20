# Quickstart

## 1. Synthetic demo (no weights, no video)

```bash
python examples/synthetic_session.py
```

Expected output:

```text
alpha range 0.0 1.0
labels {'Locomote': ..., 'Rear': ..., 'FaceGroom': ..., 'BodyGroom': ..., 'background': ...}
grooming {'total_grooming_s': ..., 'n_bouts': ..., ...}
switches {'face_to_body': ..., 'body_to_face': ..., 'total': ...}
wrote runs/synthetic/ethogram.csv
```

This exercises SGATF and the ethogram writer only; it is not a model test.

## 2. Real inference (weights required)

```bash
python scripts/infer.py \
  --video path/to/session.mp4 \
  --checkpoint path/to/best.pt \
  --config configs/infer.yaml \
  --boxes path/to/yolo_boxes.json \
  --out runs/session1
```

Outputs in `runs/session1`:

| file | content |
|---|---|
| `<name>_probs.npy` | float32 `[n_frames, 5]`, class order Locomote, Rear, FaceGroom, BodyGroom, background |
| `<name>_labels.npy` | int16 `[n_frames]`, class index |
| `<name>_ethogram.csv` | `frame,behavior` |
| `<name>_summary.json` | frame count, class composition, mean confidence |

Add `--syllables syllables.npy` to run SGATF instead of plain thresholding.

## 3. Bout statistics for a cohort

```bash
python scripts/evaluate.py --manifest cohort.csv --config configs/evaluate.yaml --out runs/cohort
```

`cohort.csv` columns: `session, group, labels` where `labels` is a path to a
`.npy` int array (or space-separated integers).

Outputs `per_animal.csv` and, for two groups, `group_stats.csv` with exact
permutation p values and BH q values.

## 4. Figures 2F / 2G

```bash
python scripts/reproduce_figures.py --figure 2f --manifest cohort.csv --out runs/fig2f
python scripts/reproduce_figures.py --figure 2g --manifest cohort.csv --out runs/fig2g
```

`cohort.csv` for Figure 2F needs `session, day, labels`; for 2G `session, group, labels`.
