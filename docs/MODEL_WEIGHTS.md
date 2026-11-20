# Model weights

No weights are distributed with this repository.

| artefact | expected file | how to obtain |
|---|---|---|
| DeepGroomNet video model | `best.pt` (state dict with `model_state`) | request from the authors; release asset after publication |
| YOLO mouse detector | `best.pt` (Ultralytics) | train on your own arena or use the authors' checkpoint |
| TAdaConvNeXtV2 Kinetics-400 init | `tadaconvnextv2_tiny_in1k_k400_79.6.pyth` | official TAdaConv release |
| DeepLabCut SuperAnimal | model zoo download | DeepLabCut 2.3.11 |
| Keypoint-MoSeq shared model | `project/` + `checkpoint.h5` | fit with `scripts/fit_kpms.py` |

Loading is tolerant to `model_state`, `state_dict` or bare state dicts, and to
a `model.` prefix from Lightning checkpoints.

Always verify the class order (`Locomote, Rear, FaceGroom, BodyGroom,
background`) and the threshold vector when you swap weights: the checkpoint
carries both in its `cfg` entry.
