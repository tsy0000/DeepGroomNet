"""
Home-Cage 数据元数据构建（Jhuang 8 类预裁剪片段）。

命名规范见 E:/tsy/clipped_database/Clipped_dataset_file_naming_convention.txt：
  文件名 = [源视频]_[行为别名]_[序号].mpg（或 .mp4）
  行为别名映射（已按规范文件核对）：
    drink:   d
    eat:     e
    groom:   gb, g, groomback, groom
    hang:    ha
    micromovement: he, head
    rear:    r
    rest:    rs
    walk:    w
输出：outputs/meta/homecage_meta.csv
    行 = (clip, class)
class ∈ {drink, eat, groom, hang, micromovement, rear, rest, walk}
"""
import csv
import os
from collections import Counter
from pathlib import Path

SUFFIX_MAP = {
    "d": "drink", "drink": "drink",
    "e": "eat", "eat": "eat",
    "gb": "groom", "g": "groom", "groomback": "groom", "groom": "groom",
    "ha": "hang", "hang": "hang",
    "he": "micromovement", "head": "micromovement",
    "r": "rear", "rear": "rear",
    "rs": "rest", "rest": "rest",
    "w": "walk", "walk": "walk",
}


def class_from_name(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 3:
        alias = parts[-2].lower()
        if alias in SUFFIX_MAP:
            return SUFFIX_MAP[alias]
    for k, v in SUFFIX_MAP.items():
        if stem.startswith(k + "_") or stem == k:
            return v
    return "unknown"


def build_homecage_meta(root, out_csv):
    root = Path(root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*")):
            if f.suffix.lower() not in (".mpg", ".mp4", ".mpeg", ".avi"):
                continue
            cls = class_from_name(f.stem)
            rows.append({"clip": str(f), "class": cls})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip", "class"])
        w.writeheader()
        w.writerows(rows)
    print(f"[homecage_meta] {len(rows)} clips -> {out_csv}")
    print("  类别分布:", dict(Counter(r["class"] for r in rows)))
    return rows


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    build_homecage_meta(cfg["DATA"]["homecage_root"],
                        os.path.join(cfg["DATA"]["meta_root"], "homecage_meta.csv"))
