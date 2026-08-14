# DeepGroomNet

Frame-level behavioral classification of top-view single-mouse open-field videos: **Locomote · Rear · FaceGroom · BodyGroom · Background**.


## Pipeline

```bash
python scripts/01_build_metadata.py   # scan videos + one-hot labels
python scripts/02_gen_clips.py        # YOLO-crop 45-frame clips
python scripts/13_gen_flow.py         # precompute optical flow (TinyMotionNet)
python train.py                       # progressive transfer + multi-task training
python scripts/11_eval_test.py        # threshold tuning on val → report on held-out test
python infer.py --video <video> --ckpt <best_model.pt>
```

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
