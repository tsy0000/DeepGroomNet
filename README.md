# DeepGroomNet

## Overview

DeepGroomNet performs frame-wise behavioral classification on top-view mouse videos to recognize five categories of behaviors: Locomote, Rear, FaceGroom, BodyGroom and Background. Centered on a video action recognition backbone, the system integrates keypoint geometric priors and optical flow motion encoding, with a full pipeline for training, evaluation and inference.

## Highlights

- **Pretrained Video Backbone**：TAdaConvNeXtV2-based architecture trained via progressive transfer learning
- **Motion Encoding**：Multi-modal input streams including raw RGB, frame difference and optical flow
- **Keypoint Fusion**：DeepLabCut keypoints converted into heatmaps fed into a 3D CNN branch, late fused with the visual stream
- **Multi-Label Classification Head**：Per-class Sigmoid activation, class-weighted loss function, and per-class threshold tuning on validation set
- **Temporal Post-Processing**：Sequence-aware decoding and bout filtering based on valid behavior duration
- **Interpretability Tools**：Gradient/attention diagnostics during training and attention heatmap visualization

## Requirements

- Python 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8
- OpenCV, Ultralytics, NumPy, Pandas, scikit-learn
- Full dependencies listed in `requirements.txt` / `environment.yml`

## Quick Start

```bash
# 1. Generate metadata and pre-clipped video segments
python scripts/01_build_metadata.py
python scripts/02_gen_clips.py

# 2. Dataset auditing and quality inspection
python scripts/07_data_audit.py

# 3. Precompute optical flow (optional, improves motion representation)
python scripts/13_gen_flow.py

# 4. Model training 
python train.py

# 5. Run inference on target video
python infer.py --video <video.mp4> --ckpt <best_model.pt>
```

## Configuration

All hyperparameters and settings are centralized in `configs/config.yaml`：

- Dataset paths and clip sampling configurations
- Backbone & attention module parameters, toggle switches for keypoint/optical flow branches
- Loss function settings and training hyperparameters
- Post-processing filtering rules

## Project Layout

```
configs/      Configuration files
data/         Data loading & preprocessing (clip sampling, keypoint parsing, optical flow & heatmap generation)
models/       Backbone network and submodules (attention layers, keypoint fusion branch)
losses/       Custom loss functions
postprocess/  Inference post-processing logic (thresholding, temporal smoothing, bout voting)
llm/          Auxiliary analytical modules (optional, excluded from model training)
scripts/      Helper scripts for data preparation, training evaluation and visualization
train.py      Main training entry (progressive transfer learning + multi-task auxiliary loss + threshold optimization)
infer.py      Main inference entry
```

## License

See `LICENSE`.
