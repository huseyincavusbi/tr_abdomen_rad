"""Sanity-check an assembled YOLO dataset (images + labels).

Usage:
  python trainer/sanity_check.py --images ds/images --labels ds/labels --out ds/sanity_report.json --montage ds/montage
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")
NAMES = [
    "acute_appendicitis",
    "acute_cholecystitis",
    "acute_pancreatitis",
    "kidney_ureteral_stone",
    "acute_diverticulitis",
    "aaa_dissection",
]
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--montage", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=0)
    ap.add_argument("--sample-images", type=int, default=25)
    ap.add_argument("--montage-count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    report = {"splits": {}, "violations": [], "classes": {n: 0 for n in NAMES}, "boxes_total": 0}
    positive_files = {"train": [], "val": [], "test": []}

    for split in SPLITS:
        img_dir, lbl_dir = args.images / split, args.labels / split
        images = sorted(img_dir.glob("*.jpg"))
        stems = {p.stem for p in images}
        labels = sorted(lbl_dir.glob("*.txt"))
        missing = sorted(stems - {p.stem for p in labels})
        orphan = sorted({p.stem for p in labels} - stems)
        empty = 0
        boxes = 0
        for lp in labels:
            lines = [ln for ln in lp.read_text().splitlines() if ln.strip()]
            if not lines:
                empty += 1
                continue
            positive_files[split].append(lp.stem)
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    report["violations"].append(f"{split}/{lp.stem}: fields={len(parts)}")
                    continue
                try:
                    c = int(float(parts[0]))
                    x, y, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    report["violations"].append(f"{split}/{lp.stem}: non-numeric '{ln}'")
                    continue
                if not 0 <= c < len(NAMES):
                    report["violations"].append(f"{split}/{lp.stem}: class={c}")
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    report["violations"].append(f"{split}/{lp.stem}: box={parts[1:]}")
                    continue
                if x - w / 2 < -0.02 or y - h / 2 < -0.02 or x + w / 2 > 1.02 or y + h / 2 > 1.02:
                    report["violations"].append(f"{split}/{lp.stem}: out-of-frame {parts[1:]}")
                    continue
                report["classes"][NAMES[c]] += 1
                boxes += 1
        report["boxes_total"] += boxes
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing),
            "orphan_labels": len(orphan),
            "negative_images": empty,
            "positive_images": len(positive_files[split]),
            "boxes": boxes,
        }
        if missing:
            report["violations"].append(f"{split}: {len(missing)} images without labels, e.g. {missing[:5]}")
        if orphan:
            report["violations"].append(f"{split}: {len(orphan)} labels without images, e.g. {orphan[:5]}")

    sizes = Counter()
    for split in SPLITS:
        images = sorted((args.images / split).glob("*.jpg"))
        if not images:
            continue
        for p in random.sample(images, min(args.sample_images, len(images))):
            try:
                with Image.open(p) as im:
                    sizes[im.size] += 1
                    if args.image_size and im.size != (args.image_size, args.image_size):
                        report["violations"].append(f"{split}/{p.stem}: size={im.size}")
            except Exception as e:
                report["violations"].append(f"{split}/{p.stem}: unreadable ({e})")
    report["image_size_samples"] = {f"{w}x{h}": n for (w, h), n in sizes.items()}

    if args.montage:
        args.montage.mkdir(parents=True, exist_ok=True)
        pool = [s for s in positive_files["train"] if (args.images / "train" / f"{s}.jpg").exists()]
        for stem in random.sample(pool, min(args.montage_count, len(pool))):
            path = args.images / "train" / f"{stem}.jpg"
            with Image.open(path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                for ln in (args.labels / "train" / f"{stem}.txt").read_text().splitlines():
                    if not ln.strip():
                        continue
                    c, x, y, w, h = ln.split()
                    c = int(float(c))
                    cx, cy, bw, bh = (float(v) * im.width for v in (x, y, w, h))
                    box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                    draw.rectangle(box, outline=COLORS[c % len(COLORS)], width=3)
                    draw.text((box[0] + 4, box[1] + 4), NAMES[c], fill=COLORS[c % len(COLORS)])
                im.save(args.montage / f"{stem}.jpg", quality=88)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["violations"]:
        print(f"FAILED: {len(report['violations'])} violations")
        raise SystemExit(1)
    print("PASS: 0 violations")


if __name__ == "__main__":
    main()
