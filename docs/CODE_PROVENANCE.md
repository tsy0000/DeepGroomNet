# Code provenance

Source project: `D:\DeepGroomNet_v3` (read-only reference). Every row records the
original file, its SHA256, what it does, the figures it feeds, whether it was verified
in this release, and where it lives in this repository.

| original file | sha256 | function | figures | verified | release path |
|---|---|---|---|---|---|
| `scripts/cv5/common.py` | `b73064d5ad15c21a…` | model assembly, checkpoint loading, sliding-window inference, thresholds | 1D-1G, 2B, 2G | yes | `deepgroomnet/models, inference, postprocessing` |
| `scripts/train_resume.py` | `67d9828cfec5f38d…` | training loop, LP/LP-FT schedule, checkpoint selection | 1D, 1E | partially (adapted to scripts/train.py) | `scripts/train.py` |
| `data/dataset.py` | `8b23256645cccbf3…` | FrameLevelDataset, 16/32-frame windows, YOLO crop, class weights | 1D, 1E, 2B | yes | `deepgroomnet/data/dataset.py` |
| `data/dataloaders.py` | `a8270f92da4e9851…` | class-balanced sampler and loader factory | 1D, 1E | yes | `deepgroomnet/data/dataloaders.py` |
| `models/tadaconv_model.py` | `8e768730f516f83a…` | TAdaConvNeXtV2 + STSA + CPAM + dual heads | 1A-1E, 2D-2G | yes (import + forward shape tests) | `deepgroomnet/models/tadaconv_model.py` |
| `models/modules/stsa.py` | `994614eafc809f4e…` | spatio-temporal shift attention after stage 3/4 | 1D, 1E | yes | `deepgroomnet/models/stsa.py` |
| `models/modules/cpam.py` | `4fd79c7f23eb491f…` | cross-stage auxiliary supervision head | 1D | yes | `deepgroomnet/models/cpam.py` |
| `losses/auxiliary.py` | `8b2dab124415b4a2…` | focal BCE, conditional weighting, L2-SP | 1D, 1E | yes (wired into scripts/train.py) | `deepgroomnet/losses/auxiliary.py` |
| `postprocess/thresholds.py` | `382a36f3d5b4ef1a…` | per-class thresholds, valid-background rule, bout postprocessing | 2B, 2F, 2G | yes (unit tests) | `deepgroomnet/postprocessing/thresholds.py` |
| `postprocess/metrics.py` | `0230f9d3da5fe505…` | precision/recall/F1/AUROC and threshold evaluation | 1D, 1E, 1G | yes (import) | `deepgroomnet/evaluation/metrics.py` |
| `scripts/fullsession_validation/fsvlib.py` | `c6e84267d896ac80…` | SGATF (F2) fusion: run means, alpha gate, class decisions | 1G, 2D-2G | yes (7 unit tests incl. reference match) | `deepgroomnet/fusion/sgatf.py` |
| `scripts/review_shank3/run_step1_predict.py` | `2c905dbeb1f94efd…` | DLC -> KPMS -> video model -> SGATF -> labels (end-to-end inference) | 2B, 2G | partially (paths parameterised; DLC/KPMS BLOCKED here) | `scripts/infer.py` |
| `scripts/review_shank3/compute_microstructure.py` | `0417379c7d7ff8cf…` | total grooming, bout count (6 s), time per bout, face/body time | 2G | yes | `deepgroomnet/evaluation/bouts.py` |
| `scripts/reviewer2_kpms/apply_shared_kpms_one.py` | `7d4a06325003ae40…` | apply a frozen KPMS model, one session | 1G, 2D-2G | no (keypoint_moseq env not installed) | `deepgroomnet/pose/kpms.py` |
| `scripts/reviewer2_kpms/kpms_fit.py` | `81bfbad1b9029735…` | KPMS PCA + AR-HMM fit, kappa scan | 1G | no (keypoint_moseq env not installed) | `deepgroomnet/pose/kpms.py` |

Full hashes

- `scripts/cv5/common.py`  `b73064d5ad15c21a21f56fef52dada910fa15311bb8b62ac396d9a95cdafc332`
- `scripts/train_resume.py`  `67d9828cfec5f38d4792d579fb1ad538034b2f3ac27c6fc0dc19248ab7c58397`
- `data/dataset.py`  `8b23256645cccbf36428f854dad476dfe9f9b10bbe9f7d96375d590a2fd40132`
- `data/dataloaders.py`  `a8270f92da4e98517916042e0e3f423582f9cc70d54dc8f7874c89cd8b803088`
- `models/tadaconv_model.py`  `8e768730f516f83a8b8151c0dbde2a7c26f452f909818b7abfbf5e291e86146f`
- `models/modules/stsa.py`  `994614eafc809f4e239e4cfa31b34ba05981d232c29b925d70191befe17f6f3c`
- `models/modules/cpam.py`  `4fd79c7f23eb491f4d5662fcabc936ec0d682dd7325fb58282b54bb3455f90dc`
- `losses/auxiliary.py`  `8b2dab124415b4a25269d1ec6b53e5ba989e490bea19b776c1a8437ce95bbbf6`
- `postprocess/thresholds.py`  `382a36f3d5b4ef1a417a1ab60d7428da71bb8a79845383840f2552aaf810a865`
- `postprocess/metrics.py`  `0230f9d3da5fe505fd3b992f2afad665cb094fdc39024e302c1504f7f1c4cfb2`
- `scripts/fullsession_validation/fsvlib.py`  `c6e84267d896ac803aba968d28294c9498f3e2d4c8b5c2c96e8cff08406229d6`
- `scripts/review_shank3/run_step1_predict.py`  `2c905dbeb1f94efdc1a63459779c417e4a82fc5610fef6a3ac42bb3182fe5083`
- `scripts/review_shank3/compute_microstructure.py`  `0417379c7d7ff8cf1bdec14f9a1f5cea596d35500d99ccc80139228484835bbd`
- `scripts/reviewer2_kpms/apply_shared_kpms_one.py`  `7d4a06325003ae4022b4eb0780c99dca566fd91fd9d856d031076ec1e9332723`
- `scripts/reviewer2_kpms/kpms_fit.py`  `81bfbad1b9029735341a7673e0cc0f72534f000dee851c69b011fe6722b6f9c6`

## Verification labels

- yes: executed in this release (imports, unit tests, or end-to-end with the shipped code).
- partially: logic carried over and re-executed, but a path/credential/weight is required
  from the user; the difference is documented in `docs/REPRODUCIBILITY.md`.
- no: source kept for reference only; the optional environment (keypoint_moseq,
  DeepLabCut) is not installed in the audit environment.

## Not migrated

Historical analysis scripts from the two review archives (figure-specific plotting,
legacy KPMS variants, diagnostic scripts) were deliberately left out. They remain in
`D:\DeepGroomNet_v3` and in the review archives, not in this repository.
