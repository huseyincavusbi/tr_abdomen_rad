"""
Automated image / label / box validation for the TR_ABDOMEN_RAD YOLO pipeline.

Checks:
  1. label coordinate validity (range, class ids, degenerate boxes)
  2. label vs ground-truth annotations (source-aware), incl. missing/extra/foreign
  3. cross-source contamination
  4. image readability / size / channel sanity
  5. DICOM -> JPEG conversion fidelity
  6. kidney-stone hyperdensity physics check
  7. patient-level split leakage
  8. overlay montages (GT box on converted image and on raw DICOM)

Usage:
  python validate_dataset.py --quick
  python validate_dataset.py --samples 4 --conversion-samples 60
"""

import argparse, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    IMAGES_DIR, LABELS_DIR, SPLITS_DIR, OUTPUT, ANNOTATIONS_DIR, CLASS_NAMES, NUM_CLASSES,
    HU_WINDOWS, IMGSZ, RANDOM_SEED,
)
from convert_dicom import find_case_dir, list_dicoms

BOX_TOL = 5e-4


def parse_prefix(name):
    parts = name.split("_")
    source = parts[0]
    case = int(parts[1])
    seri = int(parts[2][1:]) if len(parts) > 2 else None
    return source, case, seri


def fmt_prefix(source, case, seri):
    return f"{source}_{case}" if seri is None else f"{source}_{case}_s{seri}"


def iid_str(v):
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def load_native_dims(prefixes, cache_path, rebuild=False):
    if cache_path.exists() and not rebuild:
        data = json.load(open(cache_path))
        missing = [p for p in prefixes if p not in data]
        if not missing:
            return {k: tuple(v) for k, v in data.items()}
    dims = json.load(open(cache_path)) if cache_path.exists() else {}
    for p in prefixes:
        if p in dims:
            continue
        source, case, seri = parse_prefix(p)
        d = find_case_dir(source, case, seri)
        if d is None:
            continue
        fs = list_dicoms(d)
        if not fs:
            continue
        try:
            ds = pydicom.dcmread(str(fs[0]), stop_before_pixels=True)
            dims[p] = [int(ds.Rows), int(ds.Columns)]
        except Exception:
            continue
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dims, open(cache_path, "w"))
    return {k: tuple(v) for k, v in dims.items()}


def hu_from_dicom(path):
    ds = pydicom.dcmread(str(path))
    px = ds.pixel_array
    if px.ndim == 3:
        px = px[0]
    px = px.astype(np.float32)
    return px * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))


def window_rgb_bgr(hu):
    chans = []
    for center, width in HU_WINDOWS:
        lo, hi = center - width / 2.0, center + width / 2.0
        a = np.clip(hu, lo, hi)
        a = (a - lo) / (hi - lo) * 255.0
        chans.append(a.astype(np.uint8))
    rgb = np.stack(chans, axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_box(x1, y1, x2, y2, nH, nW):
    cx = (x1 + x2) / 2.0 / nW
    cy = (y1 + y2) / 2.0 / nH
    w = (x2 - x1) / nW
    h = (y2 - y1) / nH
    return (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


def parse_yolo_label(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        boxes.append((int(parts[0]), round(float(parts[1]), 6), round(float(parts[2]), 6),
                      round(float(parts[3]), 6), round(float(parts[4]), 6)))
    return boxes


def build_expected(bb, native_dims):
    exp = defaultdict(set)
    by_iid = defaultdict(list)
    for _, r in bb.iterrows():
        if float(r.x2) <= float(r.x1) or float(r.y2) <= float(r.y1):
            continue
        seri = None if pd.isna(r.get("seri")) else int(r["seri"])
        p = fmt_prefix(r["source"], int(r["case"]), seri)
        if p not in native_dims:
            continue
        nH, nW = native_dims[p]
        box = normalize_box(float(r.x1), float(r.y1), float(r.x2), float(r.y2), nH, nW)
        key = (r["source"], int(r["case"]), seri, iid_str(r["image_id"]))
        exp[key].add((int(r["class_id"]),) + box)
        by_iid[iid_str(r["image_id"])].append((key, (int(r["class_id"]),) + box))
    return exp, by_iid


def box_close(a, b, tol=BOX_TOL):
    return a[0] == b[0] and all(abs(x - y) <= tol for x, y in zip(a[1:], b[1:]))


def find_foreign(box, iid, own_key, by_iid):
    for key, bx in by_iid.get(iid, []):
        if key == own_key:
            continue
        if box_close(box, bx):
            return key
    return None


def valid_box(b):
    _, cx, cy, w, h = b
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6:
        return False
    if cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
        return False
    return True


def check_labels(bb, native_dims, label_dirs, images_dir, limit_cases=None):
    exp, by_iid = build_expected(bb, native_dims)
    res = Counter()
    examples = defaultdict(list)
    if limit_cases:
        label_dirs = label_dirs[:limit_cases]
    for d in label_dirs:
        prefix = d.name
        if prefix not in native_dims:
            res["dirs_without_native_dims"] += 1
            continue
        source, case, seri = parse_prefix(prefix)
        img_dir = images_dir / prefix
        existing = {f.stem for f in img_dir.glob("*.jpg")} if img_dir.exists() else set()
        for f in d.glob("*.txt"):
            iid = f.stem
            if iid not in existing:
                res["labels_without_image"] += 1
                continue
            produced = parse_yolo_label(f)
            own = (source, case, seri, iid)
            expected = exp.get(own, set())
            res["label_files"] += 1
            if produced:
                res["positive_files"] += 1
            else:
                res["empty_files"] += 1

            for b in produced:
                if not valid_box(b):
                    res["invalid_boxes"] += 1
                    if len(examples["invalid_boxes"]) < 5:
                        examples["invalid_boxes"].append((str(f), b))
                if not (0 <= b[0] < NUM_CLASSES):
                    res["bad_class_ids"] += 1

            for e in expected:
                if not any(box_close(b, e) for b in produced):
                    res["missing_gt_boxes"] += 1
                    if len(examples["missing"]) < 5:
                        examples["missing"].append((str(f), e))
            for b in produced:
                if not any(box_close(b, e) for e in expected):
                    foreign = find_foreign(b, iid, own, by_iid)
                    if foreign:
                        res["foreign_boxes"] += 1
                        if len(examples["foreign"]) < 5:
                            examples["foreign"].append((str(f), b, foreign))
                    else:
                        res["extra_boxes"] += 1
                        if len(examples["extra"]) < 5:
                            examples["extra"].append((str(f), b))
        res["case_dirs"] += 1
    return res, examples


def check_images(image_dirs, limit=200, seed=RANDOM_SEED):
    rng = random.Random(seed)
    res = Counter()
    images = []
    for d in image_dirs:
        images.extend(list(d.glob("*.jpg")))
    rng.shuffle(images)
    for p in images[:limit]:
        im = cv2.imread(str(p))
        if im is None:
            res["unreadable"] += 1
            continue
        if im.shape[:2] != (IMGSZ, IMGSZ) or im.shape[2] != 3:
            res["wrong_shape"] += 1
            continue
        if float(im.std()) < 1.0:
            res["flat_images"] += 1
        res["checked"] += 1
    return res


def check_conversion(bb, native_dims, n=50, seed=RANDOM_SEED):
    rng = random.Random(seed)
    rows = bb.sample(min(n, len(bb)), random_state=seed)
    diffs = []
    missing = 0
    for _, r in rows.iterrows():
        seri = None if pd.isna(r.get("seri")) else int(r["seri"])
        p = fmt_prefix(r["source"], int(r["case"]), seri)
        d = find_case_dir(r["source"], int(r["case"]), seri)
        if d is None:
            missing += 1
            continue
        dcm = Path(d) / f"{iid_str(r['image_id'])}.dcm"
        jpg = IMAGES_DIR / p / f"{iid_str(r['image_id'])}.jpg"
        if not dcm.exists() or not jpg.exists():
            missing += 1
            continue
        hu = hu_from_dicom(dcm)
        ref = cv2.resize(window_rgb_bgr(hu), (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
        got = cv2.imread(str(jpg))
        if got is None or ref.shape != got.shape:
            missing += 1
            continue
        diffs.append(float(np.mean(np.abs(ref.astype(np.float32) - got.astype(np.float32)))))
    out = Counter()
    out["checked"] = len(diffs)
    out["missing"] = missing
    if diffs:
        diffs = np.array(diffs)
        out["mad_median"] = round(float(np.median(diffs)), 3)
        out["mad_p95"] = round(float(np.percentile(diffs, 95)), 3)
        out["mad_max"] = round(float(diffs.max()), 3)
        out["over_tol"] = int((diffs > 8.0).sum())
    return out


def check_physics(n=150, seed=RANDOM_SEED):
    bb = pd.read_csv(ANNOTATIONS_DIR / "annotations_bb.csv")
    stones = bb[bb.class_name == "kidney_ureteral_stone"]
    if len(stones) == 0:
        return Counter()
    stones = stones.sample(min(n, len(stones)), random_state=seed)
    diffs = []
    miss = 0
    for _, r in stones.iterrows():
        seri = None if pd.isna(r.get("seri")) else int(r["seri"])
        d = find_case_dir(r["source"], int(r["case"]), seri)
        if d is None:
            miss += 1
            continue
        p = Path(d) / f"{iid_str(r['image_id'])}.dcm"
        if not p.exists():
            miss += 1
            continue
        a = hu_from_dicom(p)
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        pad = 8
        y0, x0 = max(0, y1 - pad), max(0, x1 - pad)
        outer = a[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        inner = a[y1:y2, x1:x2]
        if inner.size == 0 or outer.size == 0:
            continue
        mask = np.ones(outer.shape, bool)
        mask[y1 - y0:y1 - y0 + inner.shape[0], x1 - x0:x1 - x0 + inner.shape[1]] = False
        ring = outer[mask]
        if ring.size == 0:
            continue
        diffs.append(float(inner.mean() - ring.mean()))
    out = Counter()
    out["sampled"] = len(diffs)
    out["missing_dicom"] = miss
    if diffs:
        diffs = np.array(diffs)
        out["median_delta_hu"] = round(float(np.median(diffs)), 1)
        out["frac_gt_30hu"] = round(float((diffs > 30).mean()), 3)
    return out


def check_splits(prefixes, splits_dir=SPLITS_DIR):
    res = Counter()
    split_sets = {}
    for name in ["train", "val", "test"]:
        f = splits_dir / f"{name}_patients.txt"
        if not f.exists():
            res[f"{name}_missing"] += 1
            continue
        s = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
        split_sets[name] = s
        res[f"{name}_patients"] = len(s)
    names = list(split_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = split_sets[names[i]] & split_sets[names[j]]
            if ov:
                res["leakage_pairs"] += 1
                res[f"overlap_{names[i]}_{names[j]}"] = len(ov)
    cover = set().union(*split_sets.values()) if split_sets else set()

    def patient_of(p):
        parts = p.split("_")
        return f"{parts[0]}_{parts[1]}"

    res["unassigned_prefixes"] = len({p for p in prefixes if patient_of(p) not in cover})
    return res


def make_overlays(bb, native_dims, out_dir, per_class=3, seed=RANDOM_SEED):
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        rows = bb[bb.class_name == cls]
        if len(rows) == 0:
            continue
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        tiles = []
        for i in idx:
            r = rows.iloc[i]
            seri = None if pd.isna(r.get("seri")) else int(r["seri"])
            p = fmt_prefix(r["source"], int(r["case"]), seri)
            if p not in native_dims:
                continue
            jpg = IMAGES_DIR / p / f"{iid_str(r['image_id'])}.jpg"
            if not jpg.exists():
                continue
            nH, nW = native_dims[p]
            img = cv2.imread(str(jpg))
            s = IMGSZ
            x1, y1 = int(r.x1 / nW * s), int(r.y1 / nH * s)
            x2, y2 = int(r.x2 / nW * s), int(r.y2 / nH * s)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{cls} {nH}x{nW}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            tiles.append(cv2.resize(img, (400, 400)))
            if len(tiles) >= per_class:
                break
        if tiles:
            cv2.imwrite(str(out_dir / f"overlay_{cls}.png"), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip conversion/physics/overlays")
    ap.add_argument("--samples", type=int, default=3, help="overlay samples per class")
    ap.add_argument("--conversion-samples", type=int, default=50)
    ap.add_argument("--image-samples", type=int, default=200)
    ap.add_argument("--limit-cases", type=int, default=None, help="limit label dirs scanned")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT / "validation")
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR / "annotations_bb.csv")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bb = pd.read_csv(args.annotations)
    label_dirs = sorted([d for d in args.labels_dir.iterdir() if d.is_dir()])
    image_dirs = sorted([d for d in args.images_dir.iterdir() if d.is_dir()])
    prefixes = [d.name for d in label_dirs]

    print(f"annotations: {len(bb)} boxes")
    print(f"label dirs: {len(label_dirs)} | image dirs: {len(image_dirs)}")

    native_dims = load_native_dims(prefixes, args.out_dir / "native_dims.json")
    print(f"native dims known for {len(native_dims)} prefixes "
          f"({sum(1 for v in native_dims.values() if tuple(v)==(512,512))} are 512x512)")

    print("\n[1/6] labels vs ground truth ...")
    label_res, examples = check_labels(bb, native_dims, label_dirs, args.images_dir, args.limit_cases)

    print("[2/6] splits ...")
    split_res = check_splits(prefixes, args.splits_dir)

    print("[3/6] images ...")
    img_res = check_images(image_dirs, args.image_samples)

    report = {
        "labels": dict(label_res),
        "splits": dict(split_res),
        "images": dict(img_res),
    }

    if not args.quick:
        print("[4/6] conversion fidelity ...")
        report["conversion"] = dict(check_conversion(bb, native_dims, args.conversion_samples))
        print("[5/6] kidney-stone physics ...")
        report["physics"] = dict(check_physics())
        print("[6/6] overlays ...")
        make_overlays(bb, native_dims, args.out_dir / "overlays", args.samples)
        report["overlays_dir"] = str(args.out_dir / "overlays")

    print("\n================ REPORT ================")
    print(json.dumps(report, indent=2, default=str))
    print("\nexamples:")
    for k, v in examples.items():
        for e in v:
            print(f"  {k}: {e}")

    json.dump(report, open(args.out_dir / "report.json", "w"), indent=2, default=str)
    print(f"\nsaved: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
