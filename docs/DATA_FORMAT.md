# Data formats

## Behaviour labels

Frame-level integers `0..4` in the order `Locomote, Rear, FaceGroom, BodyGroom,
background`.

* training manifest CSV: `video, labels, mouse, source`
  (`labels` = path to `.npy` or space-separated integers; `source` = train/val)
* evaluation manifest CSV: `session, group, labels` and optionally `day`
* exported ethogram CSV: `frame, behavior`

## Keypoints

DeepLabCut CSV with the standard three header rows (scorer, bodyparts, coords).
`deepgroomnet.pose.dlc.keypoints_to_array` returns `[n_frames, 3 * n_bodyparts]`
padded or truncated to the video length.

## Syllables

One `int32` value per frame, saved as `.npy` with the same length as the video.
Length mismatch raises instead of silently truncating.

## YOLO boxes

JSON keyed by video stem:

```json
{"session1": {"stride": 30, "boxes": [[x1, y1, x2, y2], ...]}}
```

`stride` is the sampling interval in frames; the nearest box is used for each
frame and expanded by `margin` before the square crop.
