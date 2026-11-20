# Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: ... checkpoint` | no weights shipped | pass `--checkpoint` or set `MODEL.checkpoint` |
| all frames background | thresholds too high for your domain | calibrate thresholds with `evaluate_thresholds` on labelled validation frames |
| shape error in `sgatf` | syllable array length differs from video length | re-run `apply_kpms`, or crop to the minimum common length |
| `RuntimeError: keypoint_moseq is required` | KPMS not installed | install in a separate environment, or pass precomputed `.npy` syllables |
| `RuntimeError: DeepLabCut is required` | DLC not installed | same |
| segfault when importing torch, sklearn and cv2 together | conflicting MKL/OpenMP builds | import order is already kept lazy; recreate the environment from `docs/INSTALL.md` |
| CUDA out of memory | batch too large for 32 x 224 px clips | lower `INFERENCE.batch_size` or use `--device cpu` |
| GUI cannot find DLC/KPMS | environments not configured | set `DLC_ENV_PY`, `KPMS_ENV_PY`, `YOLO_WEIGHTS` |
| YOLO boxes missing | no `--boxes` and `yolo_crop` disabled | run `deepgroomnet.preprocessing.detector.run_yolo_detector` or disable cropping |

Test the installation with:

```bash
python -m compileall deepgroomnet scripts
pytest -q
python examples/synthetic_session.py
```
