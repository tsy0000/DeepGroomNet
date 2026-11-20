# Reproducibility

Audit date: 2026-11-20. Environment: Python 3.10.20, torch 2.6.0+cu124,
numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, opencv 5.0.0, PyYAML 6.0.3,
ultralytics 8.4.120, streamlit 1.61.1. DeepLabCut and Keypoint-MoSeq are not
installed in this environment.

| # | check | command | result |
|---|---|---|---|
| 1 | syntax / import | `python -m compileall deepgroomnet scripts` + per-module imports | PASS |
| 2 | SGATF unit tests | `pytest tests/test_sgatf.py` | PASS (7 tests) |
| 3 | model I/O shapes | `pytest tests/test_model.py` | PASS ((1,3,32,224,224) -> (1,5) + aux (1,5)) |
| 4 | class order and thresholds | `pytest tests/test_labels.py` | PASS |
| 5 | KPMS alignment / length | `pytest tests/test_sgatf.py -k length` and `probs_to_labels` length checks | PASS for the length guard; KPMS itself BLOCKED (keypoint_moseq absent) |
| 6 | statistics on saved arrays | `pytest tests/test_stats.py`, `tests/test_bouts.py` | PASS (924-split exact p = 2/924, BH monotone, 6 s merge) |
| 7 | dependency closure of train/infer/figure scripts | `python -m compileall`, import of each script module | PASS for syntax and imports; real execution needs weights/data |
| 8 | README quickstart | `python examples/synthetic_session.py` | PASS (prints alpha range, label counts, grooming block, switches) |
| 9 | personal data / credentials / absolute paths / large files | regex scan, size scan | PASS; one intentional path in `docs/CODE_PROVENANCE.md`, weights/ is git-ignored (175 MB pyth stays out) |
| 10 | third-party licences | manual audit | PARTIAL: TAdaConv LICENSE file missing in the vendored copy, upstream terms must be attached before a public tag |

Full test run: `24 passed in 10.89s` (`pytest -q`).

## What is verified end to end

* SGATF arithmetic against an independent reference implementation (allclose 1e-12).
* Model construction and forward pass for the four STSA/CPAM ablation configurations.
* Class order, valid-background rule and ethogram writer.
* Bout segmentation (5-frame merge, 6 s grooming merge) and face-body switch counting.
* Exact permutation, BH-FDR, bootstrap CI, paired test selection.

## What is PARTIAL

* `scripts/train.py`: executes and compiles, but no training dataset is shipped; the
  loop was not run to convergence here.
* `scripts/infer.py`: needs a checkpoint and a video; both are absent. The code path
  is exercised only with synthetic arrays through `Segmenter.probs_from_frames` in
  the unit tests' helper functions.
* Figure regeneration (`scripts/reproduce_figures.py`): requires per-session label
  `.npy` files. The script is syntax- and import-checked, not run on the paper cohort.
* GUI: source included, streamlit present, but the DeepLabCut/KPMS environments the
  GUI shells out to are absent, so the pose tab cannot run here.

## What is BLOCKED

* DeepLabCut 2.3.11 SuperAnimal keypoints (not installed).
* Keypoint-MoSeq 0.6.1 fit/apply (not installed).
* Model weights (not distributed): no end-to-end video inference on real data.
* Paper cohorts (videos, labels) are not in this repository.

No figure number was reproduced from model outputs in this audit. Statistics were
recomputed from arrays, which is not the same as rerunning the network.
