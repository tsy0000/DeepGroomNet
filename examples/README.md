# Examples

`synthetic_session.py` builds a random probability matrix and a syllable
sequence, runs SGATF, writes an ethogram and prints bout statistics. It checks
the API and the arithmetic, not the network.

With weights available, the end-to-end path is:

```bash
python scripts/infer.py --video session.mp4 --checkpoint best.pt \
    --boxes boxes.json --syllables syllables.npy --out runs/session1
```

Expected console summary:

```json
{"video": "...", "frames": 18299, "classes": ["Locomote", "Rear", "FaceGroom", "BodyGroom", "background"],
 "composition": {"Locomote": 0.23, "Rear": 0.05, "FaceGroom": 0.13, "BodyGroom": 0.09, "background": 0.50},
 "sgatf": true, "mean_confidence": 0.72}
```
