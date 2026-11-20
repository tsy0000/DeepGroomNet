# Third-party notices

| component | location | origin | licence |
|---|---|---|---|
| TAdaConvNeXtV2 backbone | `tadaconv/` | https://github.com/alibaba-mmai-research/TAdaConv (Alibaba Group) | Apache-2.0 (upstream); the vendored copy carries the original copyright headers |
| STSA | `deepgroomnet/models/stsa.py` | adapted from the Swin-Panda / Video-Swin pipeline described in the paper | MIT for this adaptation; upstream terms apply to derived code |
| CPAM | `deepgroomnet/models/cpam.py` | adapted from the same pipeline | MIT for this adaptation |
| DeepLabCut | optional dependency | https://github.com/DeepLabCut/DeepLabCut | LGPL-3.0 |
| Keypoint-MoSeq | optional dependency | https://github.com/dattalab/keypoint-moseq | MIT |
| Ultralytics YOLO | optional dependency | https://github.com/ultralytics/ultralytics | AGPL-3.0 |
| Streamlit GUI | `gui/app.py` | this project | MIT |

Weights are not included. The TAdaConv Kinetics-400 initialisation and any
DeepLabCut SuperAnimal weights remain under their original upstream licences.
If you redistribute a derived checkpoint, check the upstream terms first.

Action required before a public release tag: add the upstream Apache-2.0 text
for the vendored `tadaconv/` tree. The local copy does not ship a LICENSE file.
