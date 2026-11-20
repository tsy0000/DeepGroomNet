# Installation

DeepGroomNet has one light core and three optional stacks. The core runs on CPU.

## Core (segmentation, SGATF, evaluation)

Python 3.10 or newer.

```bash
pip install -e .
```

Verified on Windows 11 with Python 3.10.20, torch 2.6.0+cu124, numpy 2.2.6,
scipy 1.15.3, scikit-learn 1.7.2, opencv 5.0.0, PyYAML 6.0.3, ultralytics
8.4.120, streamlit 1.61.1. Continuous integration runs the unit tests on CPU
wheels (Ubuntu, Python 3.10).

## Optional: YOLO mouse detector

```bash
pip install -e ".[detector]"
```

Requires `ultralytics` 8.x and a YOLO checkpoint you supply (not distributed).

## Optional: pose branch

DeepLabCut and Keypoint-MoSeq pin conflicting dependencies. Use separate
environments and call the CLI from them:

```bash
conda create -n dlc python=3.9
conda create -n kpms python=3.9
```

* DeepLabCut 2.3.11 with the SuperAnimal-TopViewMouse weights (`deeplabcut`).
* Keypoint-MoSeq 0.6.1 (`keypoint_moseq`).

Both are optional: the core segmentation network runs without them, and
pre-computed syllable arrays can be passed to `scripts/infer.py --syllables`.

## Optional: GUI

```bash
pip install -e ".[gui]"
streamlit run gui/app.py
```

The GUI shells out to the DeepLabCut and Keypoint-MoSeq environments; set
`DLC_ENV_PY` and `KPMS_ENV_PY` to their interpreters, and `YOLO_WEIGHTS` to
your detector checkpoint.

## Hardware

* CPU: inference on short clips is possible but slow; tests run on CPU.
* GPU: any CUDA GPU with 8 GB is enough for 32-frame, 224 px inference at
  batch 8. Training used an RTX 4090 (24 GB) at batch 4.
