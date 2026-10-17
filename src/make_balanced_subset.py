"""Build a class-balanced, patient-disjoint demo subset from the fixed dataset.

Usage:
  python src/make_balanced_subset.py --src output/yolo_dataset_local --out output/subset
"""
import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from patient_eval import CLASS_NAMES, parse_name

RARITY = [4, 2, 5, 1, 0, 3]
TRAIN_QUOTA = {3: 450, 5: 450, 0: 400, 2: 400, 1: 400, 4: 300}
EVAL_QUOTA = {c: 80 for c in range(6)}


def collect(images_dir, labels_dir):
    patients = defaultdict(lambda: {"pos": [], "neg": []})
    for f in sorted(images_dir.glob("*.jpg")):
        patient = parse_name(f.stem)[1]
        lp = labels_dir / f"{f.stem}.txt"
        if lp.exists() and lp.stat().st_size > 0:
            classes = {int(ln.split()[0]) for ln in lp.read_text().splitlines() if ln.strip()}
            patients[patient]["pos"].append((f, classes))
        else:
            patients[patient]["neg"].append(f)
    return patients


def select(patients, quota, neg_ratio, rng):
    selected = set()
    for c in RARITY:
        got = 0
        for p in selected:
            got += sum(1 for _, cl in patients[p]["pos"] if c in cl)
        if got >= quota[c]:
            continue
        pool = [p for p in patients if any(c in cl for _, cl in patients[p]["pos"]) and p not in selected]
        rng.shuffle(pool)
        for p in pool:
            selected.add(p)
            got += sum(1 for _, cl in patients[p]["pos"] if c in cl)
            if got >= quota[c]:
                break
    pos = {}
    neg_pool = []
    for p in selected:
        for f, cl in patients[p]["pos"]:
            pos[f] = cl
        neg_pool.extend(patients[p]["neg"])
    rng.shuffle(neg_pool)
    neg = neg_pool[: int(len(pos) * neg_ratio)]
    return selected, pos, neg


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    real = Path(src).resolve()
    try:
        os.link(real, dst)
    except OSError:
        shutil.copy2(real, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("output/yolo_dataset_local"))
    ap.add_argument("--out", type=Path, default=Path("output/subset"))
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = {}
    for split, quota in (("train", TRAIN_QUOTA), ("val", EVAL_QUOTA), ("test", EVAL_QUOTA)):
        images_dir = args.src / "images" / split
        labels_dir = args.src / "labels" / split
        patients = collect(images_dir, labels_dir)
        selected, pos, neg = select(patients, quota, args.neg_ratio, rng)
        per_class = defaultdict(int)
        for cl in pos.values():
            for c in cl:
                per_class[c] += 1
        for f in pos:
            link(f, args.out / "images" / split / f.name)
            lp = labels_dir / f"{f.stem}.txt"
            if lp.exists():
                link(lp, args.out / "labels" / split / lp.name)
        for f in neg:
            link(f, args.out / "images" / split / f.name)
            lp = labels_dir / f"{f.stem}.txt"
            if lp.exists():
                link(lp, args.out / "labels" / split / lp.name)
        stats[split] = {
            "patients": len(selected),
            "positive": len(pos),
            "negative": len(neg),
            "per_class": {CLASS_NAMES[c]: per_class[c] for c in range(6)},
        }
        print(f"{split}: patients={len(selected)} pos={len(pos)} neg={len(neg)} "
              f"per_class={ {CLASS_NAMES[c]: per_class[c] for c in range(6)} }")

    names = ",".join(f"'{n}'" for n in CLASS_NAMES)
    (args.out / "dataset.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames: [{names}]\n"
    )
    print("subset:", args.out)


if __name__ == "__main__":
    main()
