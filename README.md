# DeepGroomNet

Frame-level behaviour segmentation of top-view mouse open-field video, with a
pose-guided temporal fusion step (SGATF) and the statistics used for the paper.

**What is DeepGroomNet?** A two-branch framework. The video branch classifies
32-frame clips into five behaviours (Locomote, Rear, FaceGroom, BodyGroom,
background). The pose branch converts DeepLabCut keypoints into unsupervised
Keypoint-MoSeq syllables. SGATF fuses the two by smoothing class probabilities
inside syllable runs.

**Conceptual contribution.** The paper separates WHAT is happening (semantic
classification) from WHEN it happens (temporal boundaries). The video model
answers WHAT; the pose model supplies WHEN; SGATF couples them without training
a new attention module and without assuming that a syllable equals a behaviour.

**WHAT and WHEN.** WHAT is the five-class probability vector per frame. WHEN is
the frame index at which the label changes, informed by syllable runs and by the
bout definitions in the evaluation layer.

**SGATF.** `alpha = min(1, 12/L) * clip((c - 0.35) / 0.50, 0, 1)`,
`Q = (1 - alpha) * P + alpha * mean(P within the syllable run)`, where `L` is the
run length, `c` the mean top-class probability inside the run. No new parameters.

**Why STSA and CPAM.** STSA inserts spatial-shift attention after stage 3 and 4
to sharpen small, fast local movements (paws, mouth) without changing the backbone
width. CPAM provides an auxiliary multi-stage supervision signal that stabilises
training on small cohorts. Both are optional and ablated in the paper.

## Install

```bash
pip install -e .
python -m compileall deepgroomnet scripts
pytest -q
```

Optional stacks (separate environments): `docs/INSTALL.md`.

## Run inference

```bash
python scripts/infer.py --video session.mp4 --checkpoint best.pt \
    --config configs/infer.yaml --boxes yolo_boxes.json --out runs/session1
```

Outputs `_probs.npy`, `_labels.npy`, `_ethogram.csv`, `_summary.json`.
Add `--syllables syllables.npy` to enable SGATF.

## Reproduce the major evaluations

| figure | entry point | inputs |
|---|---|---|
| 1D four-model ablation | `scripts/train.py` + `evaluation.metrics` | labelled clips, 4 configs |
| 1E 20-animal CV5 OOF | `scripts/train.py` (per fold) + `evaluation.auroc` | fold manifests |
| 1G KPMS-only / model-only / model+KPMS | `scripts/fit_kpms.py`, `scripts/infer.py` | keypoints + video |
| 2F bout organisation | `scripts/reproduce_figures.py --figure 2f` | `session, day, labels` manifest |
| 2G Shank3B grooming metrics | `scripts/reproduce_figures.py --figure 2g` | `session, group, labels` manifest |

No numbers are hard-coded anywhere; every statistic is recomputed from label
arrays. `docs/REPRODUCIBILITY.md` records what was re-executed here and what is
BLOCKED by missing weights or cohort data.

## Input formats

* video: mp4/avi at 30 fps, top-view, one mouse per arena
* labels: int array 0..4 in the order above
* keypoints: DeepLabCut CSV (three header rows)
* syllables: int32 `.npy`, one value per frame
* boxes: JSON `{stem: {stride, boxes}}`

Details in `docs/DATA_FORMAT.md`.

## Generated files

`*_probs.npy` (float32, frames x 5), `*_labels.npy` (int16), `*_ethogram.csv`
(frame, behavior), `*_summary.json` (composition, mean confidence). Cohort
evaluation adds `per_animal.csv` and `group_stats.csv`.

## Hardware and dependencies

CPU is enough for tests and for short clips. GPU (8 GB+) recommended for
inference at batch 8; training used a 24 GB card at batch 4. Versions:
`docs/INSTALL.md`. The core needs torch, numpy, scipy, scikit-learn, opencv,
pyyaml. DeepLabCut 2.3.11 and Keypoint-MoSeq 0.6.1 are optional.

## Common errors

`FileNotFoundError` on checkpoints (weights are not distributed), all-background
outputs (thresholds not calibrated for your camera), syllable length mismatch.
Full table: `docs/TROUBLESHOOTING.md`.

## Deployment constraints

* top-view 30 fps open field, one animal per recording
* five coarse classes only
* thresholds are cohort-specific
* pose branch needs two extra environments
* no weights in this repository; see `docs/MODEL_WEIGHTS.md`

## Maintenance

`docs/ROADMAP.md` lists containerisation, ONNX export, weight hosting with
checksums and a GPU CI job.

## Repository layout

```text
deepgroomnet/      package (models, preprocessing, pose, fusion, inference,
                   postprocessing, evaluation)
scripts/           train, infer, fit_kpms, evaluate, reproduce_figures
configs/           train / infer / evaluate
tests/             SGATF, model shapes, class order, bouts, statistics
examples/          synthetic_session.py (no weights needed)
docs/              install, quickstart, architecture, formats, weights,
                   reproducibility, troubleshooting, limitations, roadmap,
                   provenance, third-party notices
figures/           architecture diagram
gui/               Streamlit front-end (optional)
tadaconv/          vendored official TAdaConv backbone (third-party)
```

## Citation

See `CITATION.cff`. Please cite the article and this repository.

## Licence

MIT for this repository. Third-party components keep their own licences:
`docs/THIRD_PARTY_NOTICES.md`.
