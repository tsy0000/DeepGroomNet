# GUI

`gui/app.py` is the Streamlit front-end described in the paper. It wraps the
same code paths as the CLIs.

Requirements: `pip install -e ".[gui]"`, plus the DeepLabCut and Keypoint-MoSeq
environments. Configure with environment variables:

| variable | meaning |
|---|---|
| `DLC_ENV_PY` | interpreter of the DeepLabCut environment |
| `KPMS_ENV_PY` | interpreter of the Keypoint-MoSeq environment |
| `YOLO_WEIGHTS` | path to the YOLO detector checkpoint |
| `DEEPGROOMNET_CKPT` | default video-model checkpoint |

Missing pieces, stated explicitly: the GUI does not bundle weights, does not
ship a trained threshold file, and cannot run without the two optional
environments for the pose tab. Status: PARTIAL (source included, not executed in
this release audit because streamlit is not installed in the build environment).
