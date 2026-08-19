"""Detect mouse keypoints with DeepLabCut SuperAnimal (topview).

Mirrors dlctopview.ipynb: calls deeplabcut.video_inference_superanimal
with model_name='hrnet_w32' and detector_name='fasterrcnn_resnet50_fpn_v2',
video_adapt=True. Run with the deeplabcut conda environment.

Usage:
  D:/Anaconda3/envs/deeplabcut/python.exe scripts/detect_keypoints.py \
      <video_path> [supermodel] [pcutoff]
Prints the path of the produced keypoint CSV to stdout (web_app captures it).
"""
import os
import sys
import glob


def find_output_csv(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    cands = []
    # SuperAnimal writes an *_after_adapt.csv next to the video (or in labeled-data)
    for pat in (f"*{base}*after_adapt*.csv", f"{base}*_labeled*.csv", f"{base}*.csv"):
        cands += glob.glob(os.path.join(video_dir, "**", pat), recursive=True)
        cands += glob.glob(os.path.join(video_dir, pat))
    # also scan DLC labeled-data subfolders
    for ld in glob.glob(os.path.join(video_dir, "**", "labeled-data", "**"), recursive=True):
        if os.path.isdir(ld):
            cands += glob.glob(os.path.join(ld, f"{base}*.csv"))
    cands = sorted(set(cands))
    if not cands:
        return None
    # prefer the "after_adapt" output
    for c in cands:
        if "after_adapt" in c:
            return c
    return cands[0]


def main():
    if len(sys.argv) < 2:
        print("usage: detect_keypoints.py <video_path> [supermodel] [pcutoff]")
        sys.exit(1)
    video = sys.argv[1]
    supermodel = sys.argv[2] if len(sys.argv) > 2 else "superanimal_topviewmouse"
    pcutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    if not os.path.exists(video):
        print(f"ERROR: file not found: {video}")
        sys.exit(1)

    import deeplabcut
    ext = os.path.splitext(video)[1]
    deeplabcut.video_inference_superanimal(
        [video],
        supermodel,
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        videotype=ext,
        video_adapt=True,
        scale_list=[],
        pcutoff=pcutoff,
    )

    out = find_output_csv(video)
    if out:
        print(out)
    else:
        print("NO_OUTPUT")


if __name__ == "__main__":
    main()
