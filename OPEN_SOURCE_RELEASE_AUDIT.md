# Open-source release audit

Repository: https://github.com/tsy0000/DeepGroomNet
Branch: `review/open-source-reproducibility`
Audit date: 2026-11-20
Scope: protein-cell revision, Reviewer 2 items on Open-Source Implementation,
Technical Specifications, Troubleshooting and Maintenance.

## 1. What was done

The repository was rebuilt from the working project `D:\DeepGroomNet_v3` into a
single importable package plus five CLI entry points. Only code paths that the
paper actually uses were kept; the review archives were used to cross-check
results, not as a source of final code.

| old layout (main) | release layout |
|---|---|
| `models/`, `losses/`, `data/`, `inference/`, `postprocess/`, `train/` | `deepgroomnet/{models,losses,data,inference,postprocessing,evaluation,preprocessing,pose,fusion}` |
| 97 scripts including `diag_*`, `03_*`, `04_*`, `05_*` | 5 scripts: `train.py`, `infer.py`, `fit_kpms.py`, `evaluate.py`, `reproduce_figures.py` |
| `web_app.py` with hard-coded interpreters | `gui/app.py`, paths via `DLC_ENV_PY`, `KPMS_ENV_PY`, `YOLO_WEIGHTS` |
| `configs/config.yaml` (mixed historical settings) | `configs/train.yaml`, `configs/infer.yaml`, `configs/evaluate.yaml` |
| no SGATF module | `deepgroomnet/fusion/sgatf.py` extracted from `fsvlib.f2` |
| 8 ad-hoc tests | 5 test modules (24 tests) covering SGATF, shapes, labels, bouts, statistics |

## 2. Verification

| item | status | evidence |
|---|---|---|
| `python -m compileall deepgroomnet scripts` | PASS | exit 0 |
| `pytest -q` | PASS | 24 passed in 10.89 s |
| SGATF vs independent implementation | PASS | allclose 1e-12, 7 tests |
| model forward, 4 ablation configs | PASS | (1,3,32,224,224) -> (1,5); aux (1,5) when CPAM on |
| class order / thresholds / background rule | PASS | `tests/test_labels.py` |
| bout merge (5 frames, 6 s) and switches | PASS | `tests/test_bouts.py` |
| exact permutation (924 splits), BH, bootstrap | PASS | `tests/test_stats.py` |
| synthetic quickstart | PASS | `python examples/synthetic_session.py` |
| real video inference | BLOCKED | weights and videos are not distributed |
| DeepLabCut branch | BLOCKED | deeplabcut not installed in the audit environment |
| Keypoint-MoSeq branch | BLOCKED | keypoint_moseq not installed in the audit environment |
| GUI | PARTIAL | source parameterised; pose tab needs the two optional environments |
| figures 1D/1E/1G/2F/2G | PARTIAL | entry points and statistics verified; cohort data absent, so no number was produced here |

## 3. Repository hygiene

* no weights: `weights/` is git-ignored; the 175 MB `.pyth` file stays local.
* no videos, no raw or preprocessed animal data.
* no `__pycache__`, no `.idea`, no `outputs/`, no logs.
* absolute paths: only `docs/CODE_PROVENANCE.md` names the reference project on
  purpose. All code takes paths from CLI arguments, YAML or environment variables.
* credentials scan: no private keys, tokens or passwords found.

## 4. Third-party code

| component | shipped | licence status |
|---|---|---|
| TAdaConvNeXtV2 (`tadaconv/`) | vendored, unmodified | upstream Apache-2.0; the local copy has no LICENSE file, so the text must be added before a public tag |
| STSA / CPAM | re-implemented adaptation | MIT for this repository |
| DeepLabCut, Keypoint-MoSeq, Ultralytics, Streamlit | optional dependencies | not vendored; user installs them under their own licences |

Weights are not redistributed. The Kinetics-400 initialisation and any
SuperAnimal weights remain under their upstream terms.

## 5. Known gaps before a `v1.0.0` tag

1. Add the upstream Apache-2.0 LICENSE text for `tadoconv/`, or replace the
   vendored tree by a pinned dependency.
2. Publish model weights as a release asset with SHA256 checksums.
3. Run the DLC and KPMS branches once in their environments and record the
   commands in `docs/REPRODUCIBILITY.md`.
4. Execute `scripts/train.py` on a released fold manifest and attach the log.
5. Re-render Figures 2F/2G from the released label files.

## 6. Reviewer 2 mapping

| reviewer question | where it is answered |
|---|---|
| What is DeepGroomNet? | `README.md` first section |
| Conceptual contribution; WHAT/WHEN | `README.md`, `docs/MODEL_ARCHITECTURE.md` |
| What is SGATF? | `README.md`, `deepgroomnet/fusion/sgatf.py`, `docs/MODEL_ARCHITECTURE.md` |
| Why STSA and CPAM? | `README.md`, `docs/MODEL_ARCHITECTURE.md` |
| Install | `docs/INSTALL.md` |
| Run inference | `README.md`, `docs/QUICKSTART.md` |
| Reproduce evaluations | `README.md` table, `docs/REPRODUCIBILITY.md` |
| Input formats | `docs/DATA_FORMAT.md` |
| Output files | `docs/QUICKSTART.md` table |
| Hardware / dependencies | `docs/INSTALL.md` |
| Common errors / troubleshooting | `docs/TROUBLESHOOTING.md` |
| Deployment constraints | `docs/LIMITATIONS.md` |
| Maintenance roadmap | `docs/ROADMAP.md` |
