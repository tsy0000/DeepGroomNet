# DeepGroomNet

Frame-level behavioral classification of top-view single-mouse open-field videos: **Locomote · Rear · FaceGroom · BodyGroom · Background**.

## Design

| Layer | Approach | Provenance |
|---|---|---|
| Input | RGB + frame-difference + TinyMotionNet optical flow (6-ch, 32-frame clips) | DeepEthogram (eLife 2021) |
| Backbone | TAdaConvNeXtV2 (K400-pretrained, fp32, T-dimension preserved) | TAdaConv (CVPR 2022) |
| Geometry | DeepLabCut keypoints → PoseC3D-style heatmaps, late-fused with visual stream | PoseC3D (CVPR 2021) |
| Attention | Motion-stat gating with center/mask/keypoint priors | BehaveAI (PLOS Biol 2026) |
| Head | Multi-label sigmoid, class weights (neg/pos)^0.25, label smoothing, FG↔BG exclusivity | DeepEthogram |
| Transfer | Progressive stages + multi-task homecage aux + **L2-SP** against pretrained weights | Li et al. (ICML 2018) |
| Decoding | Per-class tuned thresholds + **data-driven per-class bout length** (1st percentile) + Sakamoto 3-stage filter | DeepEthogram / Sakamoto (2022) |

## Pipeline

```bash
python scripts/01_build_metadata.py   # scan videos + one-hot labels
python scripts/02_gen_clips.py        # YOLO-crop 45-frame clips
python scripts/13_gen_flow.py         # precompute optical flow (TinyMotionNet)
python train.py                       # progressive transfer + multi-task training
python scripts/11_eval_test.py        # threshold tuning on val → report on held-out test
python infer.py --video <video> --ckpt <best_model.pt>
```

## Highlights

- **Mature-baseline parity, more signal**: 224px full-color + flow + keypoints + attention vs. 128px binary-difference input of the FG/BG reference (Sakamoto 2022)
- **Numerically safe training**: fp32 throughout — AMP fp16 gradient overflow in TAdaConv routing/pose branches previously froze optimization (scaler backoff never fired)
- **Rigorous data integrity**: clip labels require window-homogeneous one-hot; frame-aligned flow; NaN guard on inputs; mouse-level stratified splits
- **Reproducible checkpoints**: per-epoch snapshots + last/best with optimizer & scaler state

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8, OpenCV, Ultralytics, NumPy, Pandas, scikit-learn, SciPy. See `requirements.txt`.

## Layout

```
configs/       central configuration
data/          clip sampling, keypoints, flow, augmentation
models/        backbone, attention, pose branch
losses/        weighted BCE, FG/BG exclusivity, L2-SP
postprocess/   thresholding, bout filtering, sequence decode
scripts/       data prep, training, evaluation, diagnostics
train.py       training entry
infer.py       inference entry
```

## Notes

- Dataset paths in `configs/config.yaml` are local; adjust before use.
- Top-view grooming datasets used for training follow the 6-column one-hot format (`background, Locomote, Rear, FaceGroom, BodyGroom, Defecate`).
