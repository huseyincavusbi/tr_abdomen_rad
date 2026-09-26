"""Build per-image anatomical gates from the dataset boundary annotations.

Each case has start/end slices per organ; a slice is "in range" for the organ
(and therefore for the class mapped to it) only within those bounds.
Used by patient_eval.py --gates to reject out-of-organ detections.

Usage:
  python src/build_slice_gates.py --boundaries output/labels/annotations_boundary.csv \
      --images output/images --out annotations/slice_gates.csv --margin-slices 0
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from patient_eval import CLASS_NAMES
from config import ANNOTATIONS_DIR

ORGAN_TO_CLASS = {
    "kidney_bladder": "kidney_ureteral_stone",
    "gall_bladder": "acute_cholecystitis",
    "appendix": "acute_appendicitis",
    "pancreas": "acute_pancreatitis",
    "colon": "acute_diverticulitis",
    "abdominal_aorta": "aaa_dissection",
}


def fmt_partition(source, case, seri):
    return f"{source}_{case}" if seri is None else f"{source}_{case}_s{seri}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundaries", type=Path, default=ANNOTATIONS_DIR / "annotations_boundary.csv")
    ap.add_argument("--images", type=Path, default=Path("output/images"))
    ap.add_argument("--out", type=Path, default=ANNOTATIONS_DIR / "slice_gates.csv")
    ap.add_argument("--margin-slices", type=int, default=0)
    args = ap.parse_args()

    bnd = pd.read_csv(args.boundaries, dtype={"image_id": str})
    bnd = bnd[bnd.organ.isin(ORGAN_TO_CLASS)]

    groups = defaultdict(list)
    for r in bnd.itertuples():
        seri = None if pd.isna(r.seri) else int(r.seri)
        groups[(fmt_partition(r.source, int(r.case), seri), r.organ)].append(
            (r.boundary_type if isinstance(r.boundary_type, str) else None, str(r.image_id))
        )

    rows = []
    stats = defaultdict(int)
    for (partition, organ), entries in groups.items():
        img_dir = args.images / partition
        if not img_dir.exists():
            stats["missing_image_dir"] += 1
            continue
        stems = sorted(f.stem for f in img_dir.glob("*.jpg"))
        pos = {s: i for i, s in enumerate(stems)}
        explicit = [e for e in entries if e[0] in ("start", "end")]
        if explicit and len(explicit) == len(entries):
            starts = sorted(pos[i] for t, i in entries if t == "start" and i in pos)
            stops = sorted(pos[i] for t, i in entries if t == "end" and i in pos)
            if not starts or len(starts) != len(stops):
                stats["unpaired_ranges"] += 1
                continue
            pairs = [(min(a, b), max(a, b)) for a, b in zip(starts, stops)]
        else:
            ps = sorted(pos[i] for _, i in entries if i in pos)
            if len(ps) < 2 or len(ps) % 2:
                stats["odd_groups"] += 1
                continue
            pairs = [(ps[k], ps[k + 1]) for k in range(0, len(ps), 2)]
        cls = CLASS_NAMES.index(ORGAN_TO_CLASS[organ])
        for lo, hi in pairs:
            lo = max(0, lo - args.margin_slices)
            hi = min(len(stems) - 1, hi + args.margin_slices)
            for s in stems[lo:hi + 1]:
                rows.append((partition, s, cls))
            stats["ranges"] += 1
        stats["partitions"] += 1

    out = pd.DataFrame(sorted(set(rows)), columns=["partition", "image_id", "class_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"gates: {len(out)} rows -> {args.out}")
    print(f"partitions={stats['partitions']} ranges={stats['ranges']} "
          f"missing_image_dir={stats['missing_image_dir']} unpaired_ranges={stats['unpaired_ranges']} "
          f"odd_groups={stats['odd_groups']}")
    print(out.class_id.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
