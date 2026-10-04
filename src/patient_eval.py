"""Patient-level evaluation for the 6-class abdominal CT detector.

Protocol (Erdogan & Koc 2026): per-patient score = aggregation of slice
confidences (max or top-k mean), per-class threshold swept on val to maximize
F1 and frozen for test, independent per-class TP/FP/TN/FN, macro-averaged.
Also: AUROC, sensitivity at target specificity, optional anatomical gating,
optional 9-region localization, optional specificity on a negative cohort,
optional bootstrap CIs.

Usage:
  python src/patient_eval.py predict --model runs/.../best.pt --images ds/images/val --out pred/val.csv
  python src/patient_eval.py eval --val-pred pred/val.csv --test-pred pred/test.csv \
      --val-images ds/images/val --test-images ds/images/test \
      --gates annotations/slice_gates.csv --score topk --topk 3 \
      --grid --bootstrap 1000 --out output/eval_patient
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES = [
    "acute_appendicitis",
    "acute_cholecystitis",
    "acute_pancreatitis",
    "kidney_ureteral_stone",
    "acute_diverticulitis",
    "aaa_dissection",
]
NUM_CLASSES = len(CLASS_NAMES)
SWEEP = np.round(np.arange(0.01, 1.0, 0.01), 2)


def parse_name(stem):
    parts = stem.split("_")
    source, case = parts[0], parts[1]
    seri = None
    if len(parts) > 3 and parts[2].startswith("s") and parts[2][1:].isdigit():
        seri = parts[2][1:]
        image_id = "_".join(parts[3:])
    else:
        image_id = "_".join(parts[2:])
    partition = f"{source}_{case}" + (f"_s{seri}" if seri else "")
    return partition, f"{source}_{case}", image_id


def collect_images(images, list_file):
    if list_file:
        return [Path(line.strip()) for line in Path(list_file).read_text().splitlines() if line.strip()]
    return sorted(Path(images).glob("*.jpg"))


def read_split_stems(images_dir):
    return {f.stem: parse_name(f.stem) for f in sorted(Path(images_dir).glob("*.jpg"))}


def read_labels(labels_dir, stems):
    positives = defaultdict(set)
    boxes = defaultdict(list)
    labels_dir = Path(labels_dir)
    for stem in stems:
        path = labels_dir / f"{stem}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            positives[stems[stem][1]].add(c)
            boxes[(stems[stem][1], c)].append((cx, cy, w, h))
    return positives, boxes


def region_of(cx, cy):
    return min(int(cx * 3), 2) * 3 + min(int(cy * 3), 2)


def load_gates(path):
    gates = defaultdict(set)
    if not path:
        return gates
    df = pd.read_csv(path, dtype={"image_id": str})
    for r in df.itertuples():
        gates[(r.partition, int(r.class_id))].add(str(r.image_id))
    return gates


def aggregate(pred_df, stems, gates, score_mode, topk):
    scores = defaultdict(list)
    pred_region = {}
    for r in pred_df.itertuples():
        meta = stems.get(str(r.image))
        if meta is None:
            continue
        partition, patient, image_id = meta
        c = int(r.class_id)
        allowed = gates.get((partition, c))
        if allowed is not None and str(image_id) not in allowed:
            continue
        scores[(patient, c)].append(float(r.conf))
        key = (patient, c)
        if key not in pred_region or r.conf > pred_region[key][0]:
            pred_region[key] = (float(r.conf), region_of((r.x1 + r.x2) / 2 / r.w, (r.y1 + r.y2) / 2 / r.h))
    out = {}
    for key, vals in scores.items():
        vals.sort(reverse=True)
        out[key] = float(np.mean(vals[:topk])) if score_mode == "topk" else float(vals[0])
    return out, {k: v[1] for k, v in pred_region.items()}


def sweep_thresholds(scores, positives, patients):
    thresholds = {}
    for c in range(NUM_CLASSES):
        y = np.array([1 if c in positives.get(p, set()) else 0 for p in patients])
        s = np.array([scores.get((p, c), 0.0) for p in patients])
        best_f1, best_t = -1.0, 0.5
        for t in SWEEP:
            pred = s >= t
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = {"threshold": best_t, "val_f1": round(best_f1, 4)}
    return thresholds


def class_metrics(scores, positives, patients, thr, c):
    y = np.array([1 if c in positives.get(p, set()) else 0 for p in patients])
    s = np.array([scores.get((p, c), 0.0) for p in patients])
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    out = {
        "positives": int(y.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": None,
        "sens_at_spec": {},
    }
    if 0 < y.sum() < len(y):
        out["auroc"] = round(float(roc_auc_score(y, s)), 4)
        fpr, tpr, _ = roc_curve(y, s)
        spec = 1 - fpr
        for target in (0.90, 0.95):
            ok = spec >= target
            out["sens_at_spec"][f"spec_{int(target * 100)}"] = round(float(tpr[ok].max()), 4) if ok.any() else None
    return out


def macro(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def specificity_at(neg_scores, patients, thresholds):
    per_class = []
    for c in range(NUM_CLASSES):
        t = thresholds[c]["threshold"]
        flagged = sum(1 for p in patients if neg_scores.get((p, c), 0.0) >= t)
        per_class.append(round(1 - flagged / max(len(patients), 1), 4))
    return per_class


def grid_localization(scores, pred_region, gt_boxes, positives, patients, thresholds):
    restricted_ok = restricted_n = covered_ok = covered_n = 0
    for p in patients:
        for c in range(NUM_CLASSES):
            if c not in positives.get(p, set()):
                continue
            regions = {region_of(cx, cy) for cx, cy, w, h in gt_boxes.get((p, c), [])}
            covered_n += 1
            if scores.get((p, c), 0.0) < thresholds[c]["threshold"]:
                continue
            restricted_n += 1
            if pred_region.get((p, c)) in regions:
                restricted_ok += 1
                covered_ok += 1
    return {
        "restricted": round(restricted_ok / max(restricted_n, 1), 4),
        "restricted_n": restricted_n,
        "coverage_inclusive": round(covered_ok / max(covered_n, 1), 4),
        "coverage_n": covered_n,
    }


def bootstrap_ci(scores, positives, patients, thresholds, n_boot, seed=0):
    if n_boot <= 0 or not patients:
        return None
    rng = np.random.default_rng(seed)
    macro_auroc, macro_f1 = [], []
    for _ in range(n_boot):
        sample = list(rng.choice(patients, size=len(patients), replace=True))
        aurocs, f1s = [], []
        for c in range(NUM_CLASSES):
            y = np.array([1 if c in positives.get(p, set()) else 0 for p in sample])
            s = np.array([scores.get((p, c), 0.0) for p in sample])
            if 0 < y.sum() < len(y):
                aurocs.append(roc_auc_score(y, s))
            pred = s >= thresholds[c]["threshold"]
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        macro_auroc.append(np.mean(aurocs) if aurocs else np.nan)
        macro_f1.append(np.mean(f1s))
    def ci(v):
        v = [x for x in v if not np.isnan(x)]
        if not v:
            return None
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"macro_auroc_ci": ci(macro_auroc), "macro_f1_ci": ci(macro_f1), "n_boot": n_boot}


def evaluate_split(pred_path, images_dir, labels_dir, gates, args, thresholds=None):
    stems = read_split_stems(images_dir)
    positives, gt_boxes = read_labels(labels_dir, stems)
    patients = sorted({v[1] for v in stems.values()})
    if args.limit_images and len(stems) > args.limit_images:
        keep = set(sorted(stems)[: args.limit_images])
        stems = {k: v for k, v in stems.items() if k in keep}
        positives, gt_boxes = read_labels(labels_dir, stems)
        patients = sorted({v[1] for v in stems.values()})
    pred = pd.read_csv(pred_path, dtype={"image": str})
    scores, pred_region = aggregate(pred, stems, gates, args.score, args.topk)
    if thresholds is None:
        thresholds = sweep_thresholds(scores, positives, patients)
    per_class = {}
    for c in range(NUM_CLASSES):
        per_class[CLASS_NAMES[c]] = class_metrics(scores, positives, patients, thresholds[c]["threshold"], c)
    result = {
        "patients": len(patients),
        "images": len(stems),
        "thresholds": {CLASS_NAMES[c]: thresholds[c] for c in range(NUM_CLASSES)},
        "per_class": per_class,
        "macro": {
            "auroc": macro([per_class[n]["auroc"] for n in CLASS_NAMES]),
            "f1": macro([per_class[n]["f1"] for n in CLASS_NAMES]),
            "precision": macro([per_class[n]["precision"] for n in CLASS_NAMES]),
            "recall": macro([per_class[n]["recall"] for n in CLASS_NAMES]),
        },
    }
    if args.grid:
        result["localization"] = grid_localization(scores, pred_region, gt_boxes, positives, patients, thresholds)
    if args.bootstrap > 0:
        result["bootstrap"] = bootstrap_ci(scores, positives, patients, thresholds, args.bootstrap)
    score_df = pd.DataFrame(
        [(p, CLASS_NAMES[c], scores.get((p, c), 0.0), int(c in positives.get(p, set()))) for p in patients for c in range(NUM_CLASSES)],
        columns=["patient", "class", "score", "label"],
    )
    return result, thresholds, score_df


def cmd_predict(args):
    from ultralytics import YOLO

    files = collect_images(args.images, args.list)
    if not files:
        raise SystemExit("no images found for prediction")
    source_list = Path(args.out).parent / f"{Path(args.out).stem}_source.txt"
    source_list.parent.mkdir(parents=True, exist_ok=True)
    source_list.write_text("\n".join(str(p) for p in files) + "\n")
    model = YOLO(args.model)
    rows = []
    results = model.predict(
        source=str(source_list),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        augment=args.augment,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )
    for path, r in zip(files, results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        h, w = r.orig_shape
        stem = path.stem
        for (x1, y1, x2, y2), k, c in zip(xyxy, confs, classes):
            rows.append((stem, int(c), float(k), float(x1), float(y1), float(x2), float(y2), int(h), int(w)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image", "class_id", "conf", "x1", "y1", "x2", "y2", "h", "w"]).to_csv(out, index=False)
    print(f"predictions: {len(rows)} boxes from {len(files)} images -> {out}")


def cmd_eval(args):
    gates = load_gates(args.gates)
    val_result, thresholds, val_scores = evaluate_split(args.val_pred, args.val_images, args.val_labels, gates, args)
    test_result, _, test_scores = evaluate_split(args.test_pred, args.test_images, args.test_labels, gates, args, thresholds)
    report = {"config": vars(args), "val": val_result, "test": test_result}
    if args.neg_pred:
        neg_stems = read_split_stems(args.neg_images)
        neg_patients = sorted({v[1] for v in neg_stems.values()})
        neg_pred = pd.read_csv(args.neg_pred, dtype={"image": str})
        neg_scores, _ = aggregate(neg_pred, neg_stems, gates, args.score, args.topk)
        spec = specificity_at(neg_scores, neg_patients, thresholds)
        report["specificity"] = {
            "patients": len(neg_patients),
            "per_class": {CLASS_NAMES[c]: spec[c] for c in range(NUM_CLASSES)},
            "macro": macro(spec),
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_scores.to_csv(out / "scores_val.csv", index=False)
    test_scores.to_csv(out / "scores_test.csv", index=False)
    report["config"] = {k: str(v) for k, v in report["config"].items()}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"val": {k: report["val"][k] for k in ("patients", "macro")},
                      "test": {k: report["test"][k] for k in ("patients", "macro")}}, indent=2))
    if "specificity" in report:
        print(f"specificity macro: {report['specificity']['macro']} (n={report['specificity']['patients']})")
    print(f"saved: {out / 'report.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--list", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--augment", action="store_true")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("eval")
    e.add_argument("--val-pred", type=Path, required=True)
    e.add_argument("--test-pred", type=Path, required=True)
    e.add_argument("--val-images", type=Path, required=True)
    e.add_argument("--test-images", type=Path, required=True)
    e.add_argument("--val-labels", type=Path)
    e.add_argument("--test-labels", type=Path)
    e.add_argument("--neg-pred", type=Path)
    e.add_argument("--neg-images", type=Path)
    e.add_argument("--gates", type=Path)
    e.add_argument("--score", choices=("max", "topk"), default="max")
    e.add_argument("--topk", type=int, default=3)
    e.add_argument("--limit-images", type=int, default=None)
    e.add_argument("--grid", action="store_true")
    e.add_argument("--bootstrap", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    if args.command == "eval":
        if not args.val_labels:
            args.val_labels = args.val_images.parent.parent / "labels" / args.val_images.name
        if not args.test_labels:
            args.test_labels = args.test_images.parent.parent / "labels" / args.test_images.name
    args.func(args)


if __name__ == "__main__":
    main()
