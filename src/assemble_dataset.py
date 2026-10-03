"""Assemble a YOLO dataset from abdomen-prepared images + these fixed labels.

Usage:
  python assemble_dataset.py --images /kaggle/input/abdomen-prepared/yolo_dataset/images --labels /kaggle/working/labels --out /kaggle/working/ds
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--images", type=Path, required=True)
ap.add_argument("--labels", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

if not args.images.exists():
    raise SystemExit(f"ERROR: images dir not found: {args.images}")
if not args.labels.exists():
    raise SystemExit(f"ERROR: labels dir not found: {args.labels}")

created = 0
for split in ["train", "val", "test"]:
    (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for lbl in (args.labels / split).glob("*.txt"):
        img = args.images / split / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        il = args.out / "images" / split / img.name
        ll = args.out / "labels" / split / lbl.name
        if not il.exists():
            il.symlink_to(img)
        if not ll.exists():
            ll.symlink_to(lbl)
        n += 1
    print(f"{split}: {n} pairs")
    created += n
if created == 0:
    raise SystemExit("ERROR: no image/label pairs matched")

(args.out / "dataset.yaml").write_text(
    f"path: {args.out.resolve()}\n"
    "train: images/train\nval: images/val\ntest: images/test\n"
    "nc: 6\nnames: ['acute_appendicitis','acute_cholecystitis','acute_pancreatitis',"
    "'kidney_ureteral_stone','acute_diverticulitis','aaa_dissection']\n"
)
print("dataset ready:", args.out / "dataset.yaml")
