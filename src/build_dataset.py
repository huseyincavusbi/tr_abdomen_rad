"""
Build the YOLO dataset from converted images and ground-truth annotations.

Two corrections vs build_kaggle_dataset.py / build_yolo.py:
  1. bounding boxes are normalized by each case's NATIVE (H, W), not by imgsz
  2. annotations are joined on (source, case, seri, image_id), never on case alone

Writes labels, a symlink dataset and patient-level splits, and never modifies
existing outputs.

Usage:
  python build_dataset.py
  python build_dataset.py --labels-out output/labels --yolo-out output/yolo_dataset_local
"""

import argparse, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ANNOTATIONS_DIR, IMAGES_DIR, OUTPUT, CLASS_NAMES, NUM_CLASSES,
    NEG_POS_RATIO, SPLIT_RATIO, RANDOM_SEED,
)
from convert_dicom import find_case_dir, list_dicoms


def parse_prefix(name):
    parts = name.split("_")
    return parts[0], int(parts[1]), (int(parts[2][1:]) if len(parts) > 2 else None)


def fmt_prefix(source, case, seri):
    return f"{source}_{case}" if seri is None else f"{source}_{case}_s{seri}"


def iid_str(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def normalize_box(x1, y1, x2, y2, nH, nW):
    cx = max(0.0, min(1.0, (x1 + x2) / 2.0 / nW))
    cy = max(0.0, min(1.0, (y1 + y2) / 2.0 / nH))
    w = max(0.001, min(1.0, (x2 - x1) / nW))
    h = max(0.001, min(1.0, (y2 - y1) / nH))
    return cx, cy, w, h


def load_native_dims(prefixes, cache_path):
    data = json.load(open(cache_path)) if cache_path.exists() else {}
    added = 0
    for p in prefixes:
        if p in data:
            continue
        source, case, seri = parse_prefix(p)
        d = find_case_dir(source, case, seri)
        if d is None:
            continue
        fs = list_dicoms(d)
        if not fs:
            continue
        try:
            import pydicom
            ds = pydicom.dcmread(str(fs[0]), stop_before_pixels=True)
            data[p] = [int(ds.Rows), int(ds.Columns)]
            added += 1
        except Exception:
            continue
    if added:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(cache_path, "w"))
    return {k: tuple(v) for k, v in data.items()}


def write_yolo_label(path, boxes):
    lines = []
    for cid, x1, y1, x2, y2, nH, nW in boxes:
        cx, cy, w, h = normalize_box(x1, y1, x2, y2, nH, nW)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-out", type=Path, default=OUTPUT / "labels")
    ap.add_argument("--yolo-out", type=Path, default=OUTPUT / "yolo_dataset_local")
    ap.add_argument("--splits-out", type=Path, default=OUTPUT / "splits")
    ap.add_argument("--native-dims", type=Path, default=OUTPUT / "validation" / "native_dims.json")
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR / "annotations_bb.csv")
    args = ap.parse_args()

    random.seed(RANDOM_SEED)
    bb = pd.read_csv(args.annotations)
    print(f"annotations: {len(bb)} boxes, {bb['case'].nunique()} case numbers")

    image_dirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir() and d.name[0].isalpha()])
    prefixes = [d.name for d in image_dirs]
    print(f"converted case dirs: {len(prefixes)}")

    native_dims = load_native_dims(prefixes, args.native_dims)
    print(f"native dims: {len(native_dims)} prefixes")

    ann_index = defaultdict(lambda: defaultdict(list))
    for _, r in bb.iterrows():
        seri = None if pd.isna(r.get("seri")) else int(r["seri"])
        key = (r["source"], int(r["case"]), seri)
        ann_index[key][iid_str(r["image_id"])].append(
            (int(r["class_id"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        )

    args.labels_out.mkdir(parents=True, exist_ok=True)
    patient_slices = defaultdict(list)
    patient_classes = defaultdict(set)
    total_slices = 0
    positives = 0
    no_native = 0

    for prefix in prefixes:
        source, case, seri = parse_prefix(prefix)
        if prefix not in native_dims:
            no_native += 1
            continue
        nH, nW = native_dims[prefix]
        patient = f"{source}_{case}"
        img_dir = IMAGES_DIR / prefix
        anns = ann_index.get((source, case, seri), {})

        for jpg in sorted(img_dir.glob("*.jpg")):
            iid = jpg.stem
            boxes = [(cid, x1, y1, x2, y2, nH, nW) for cid, x1, y1, x2, y2 in anns.get(iid, [])
                     if x2 > x1 and y2 > y1]
            write_yolo_label(args.labels_out / prefix / f"{iid}.txt", boxes)
            key = (prefix, iid)
            patient_slices[patient].append(key)
            total_slices += 1
            if boxes:
                positives += 1
                for cid, *_ in boxes:
                    patient_classes[patient].add(cid)

    print(f"labels written: {total_slices} slices, {positives} positive, "
          f"{total_slices - positives} negative ({no_native} dirs skipped)")

    patients = sorted(patient_classes.keys())
    primary_class = {p: min(patient_classes[p]) for p in patients}
    print(f"annotated patients: {len(patients)}")

    from sklearn.model_selection import train_test_split
    stratify = [primary_class[p] for p in patients]
    counts = Counter(stratify)
    if all(v >= 2 for v in counts.values()) and len(patients) >= 6:
        train_p, holdout_p = train_test_split(
            patients, test_size=SPLIT_RATIO[1] + SPLIT_RATIO[2],
            random_state=RANDOM_SEED, stratify=stratify)
        temp_strat = [primary_class[p] for p in holdout_p]
        if all(v >= 2 for v in Counter(temp_strat).values()) and len(holdout_p) >= 4:
            val_p, test_p = train_test_split(
                holdout_p, test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]),
                random_state=RANDOM_SEED, stratify=temp_strat)
        else:
            val_p, test_p = train_test_split(
                holdout_p, test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]),
                random_state=RANDOM_SEED)
    else:
        print("[WARN] stratification not feasible; random split")
        train_p, holdout_p = train_test_split(
            patients, test_size=SPLIT_RATIO[1] + SPLIT_RATIO[2], random_state=RANDOM_SEED)
        val_p, test_p = train_test_split(
            holdout_p, test_size=SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2]),
            random_state=RANDOM_SEED)

    print(f"split patients: train {len(train_p)}, val {len(val_p)}, test {len(test_p)}")

    def slices_of(plist, sample_neg=None):
        keys = []
        for p in plist:
            keys.extend(patient_slices.get(p, []))
        pos, neg = [], []
        for k in keys:
            if (args.labels_out / k[0] / f"{k[1]}.txt").stat().st_size > 0:
                pos.append(k)
            else:
                neg.append(k)
        if sample_neg is not None and len(neg) > sample_neg:
            neg = random.sample(neg, sample_neg)
        return pos, neg

    train_pos, train_neg_sample = slices_of(train_p, sample_neg=None)
    target_neg = min(len(train_neg_sample), len(train_pos) * NEG_POS_RATIO)
    train_neg = random.sample(train_neg_sample, target_neg) if len(train_neg_sample) > target_neg else train_neg_sample
    val_pos, val_neg = slices_of(val_p)
    test_pos, test_neg = slices_of(test_p)

    print(f"train: {len(train_pos)} pos + {len(train_neg)} neg | "
          f"val: {len(val_pos) + len(val_neg)} | test: {len(test_pos) + len(test_neg)}")

    yolo = args.yolo_out
    for split, keys in [("train", train_pos + train_neg), ("val", val_pos + val_neg), ("test", test_pos + test_neg)]:
        img_dst = yolo / "images" / split
        lbl_dst = yolo / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for prefix, iid in keys:
            flat = f"{prefix}_{iid}"
            img_link = img_dst / f"{flat}.jpg"
            lbl_link = lbl_dst / f"{flat}.txt"
            if not img_link.exists():
                img_link.symlink_to(IMAGES_DIR / prefix / f"{iid}.jpg")
            if not lbl_link.exists():
                lbl_link.symlink_to(args.labels_out / prefix / f"{iid}.txt")

    args.splits_out.mkdir(parents=True, exist_ok=True)
    for name, plist in [("train", train_p), ("val", val_p), ("test", test_p)]:
        (args.splits_out / f"{name}_patients.txt").write_text("\n".join(plist) + "\n")

    yaml_path = yolo / "dataset.yaml"
    yaml_path.write_text(
        f"path: {yolo}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"\ndataset.yaml: {yaml_path}")
    print(f"labels: {args.labels_out}")
    print(f"yolo:   {yolo}")
    print(f"splits: {args.splits_out}")


if __name__ == "__main__":
    main()
