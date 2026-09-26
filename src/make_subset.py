"""Build a small subset YOLO dataset (symlinks) from a full one.

Usage:
  python src/make_subset.py --src ds --out ds_mini --train 1200 --val 400 --test 200 --per-patient 2
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patient_eval import CLASS_NAMES, parse_name

NAMES = ",".join(f"'{n}'" for n in CLASS_NAMES)


def pick(files, n, per_patient, rng):
    by_patient = defaultdict(list)
    for f in files:
        by_patient[parse_name(f.stem)[1]].append(f)
    patients = sorted(by_patient)
    rng.shuffle(patients)
    chosen = []
    for p in patients:
        imgs = by_patient[p]
        rng.shuffle(imgs)
        chosen.extend(imgs[:per_patient])
        if len(chosen) >= n:
            break
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("ds"))
    ap.add_argument("--out", type=Path, default=Path("ds_mini"))
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        out_img = args.out / "images" / split
        out_lbl = args.out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        chosen = pick(sorted((args.src / "images" / split).glob("*.jpg")), n, args.per_patient, rng) if n else []
        for f in chosen:
            (out_img / f.name).symlink_to(f.resolve())
            lp = args.src / "labels" / split / f"{f.stem}.txt"
            if lp.exists():
                (out_lbl / lp.name).symlink_to(lp.resolve())
        print(f"{split}: {len(chosen)} images")
    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 6\nnames: [{NAMES}]\n"
    )
    print("dataset:", args.out / "dataset.yaml")


if __name__ == "__main__":
    main()
