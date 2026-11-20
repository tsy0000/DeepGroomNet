# Model architecture

Input: RGB clip `[3, 32, 224, 224]`, per-frame normalisation `(x - 0.45) / 0.225`.

Backbone: TAdaConvNeXtV2-Tiny (official implementation, vendored under
`tadaconv/`). Stem `Conv3d(3 -> 96, k=(3,4,4), s=(1,4,4))`, then three
downsample layers and four stages with depths `(3, 3, 9, 3)` and widths
`(96, 192, 384, 768)`.

| component | placement | input |
|---|---|---|
| STSA | after stage 3 (192) and after stage 4 (768) | stage feature map |
| CPAM | auxiliary branch over stages 1-4 | concatenated multi-stage features |
| main head | `Dropout(0.1) -> Linear(768, 5)` | globally pooled stage-4 feature |
| auxiliary head | inside CPAM | CPAM fused feature |

Outputs: five independent sigmoid probabilities (multi-label), order
`Locomote, Rear, FaceGroom, BodyGroom, background`. `background` follows the
DeepEthogram valid-background rule: it is active only when no other class
exceeds its threshold.

Loss (`deepgroomnet/losses/auxiliary.py`): focal binary cross-entropy on the
main head, focal BCE on the auxiliary head, combined by the conditional
weighting of the paper (`alpha = 0.9`, `beta = 0.1`), plus optional L2-SP.
Training and inference differ only in dropout/stochastic depth and in the
auxiliary head being unused at inference.

Thresholds are per class, selected on the validation animals (grid search
maximising F1, floor 0.01, fallback 0.5 if F1 < 0.05) and then frozen.
