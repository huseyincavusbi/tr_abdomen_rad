"""
Full parallel DICOM → 1280 JPEG conversion for all cases.
Uses multiprocessing across performance cores.
"""

import sys, os, time, warnings, multiprocessing as mp
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from config import EGITIM, YARISMA, EK, IMAGES_DIR, LABELS_DIR
from convert_dicom import convert_case
import pandas as pd


def collect_all_cases():
    """Collect all (source, case, seri) tuples to process."""
    cases = []
    
    # Egitim
    if EGITIM.exists():
        for d in sorted(EGITIM.iterdir()):
            if d.is_dir() and d.name.isdigit():
                cases.append(("egitim", int(d.name), None))
    
    # Yarisma
    if YARISMA.exists():
        for d in sorted(YARISMA.iterdir()):
            if d.is_dir() and d.name.isdigit():
                cases.append(("yarisma", int(d.name), None))
    
    # Ek — need case + seri from annotations
    bb = pd.read_csv(LABELS_DIR / "annotations_bb.csv")
    ek_cases = bb[bb["source"] == "ek"][["case", "seri"]].drop_duplicates()
    for _, row in ek_cases.iterrows():
        cases.append(("ek", int(row["case"]), int(row["seri"])))
    
    return cases


def worker(args):
    """Convert one case. Returns (prefix, n_slices, elapsed)."""
    source, case_num, seri, target_size = args
    prefix = f"{source}_{case_num}" if seri is None else f"{source}_{case_num}_s{seri}"
    t0 = time.time()
    try:
        results = convert_case(source, case_num, seri, target_size=target_size, overwrite=False, fmt="jpg")
        n = len(results) if results else 0
        dt = time.time() - t0
        return (prefix, n, dt, None)
    except Exception as e:
        dt = time.time() - t0
        return (prefix, 0, dt, str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit cases (for testing)")
    args = parser.parse_args()

    cases = collect_all_cases()
    if args.limit:
        cases = cases[:args.limit]
    
    print(f"=== Full DICOM → {args.imgsz} JPEG conversion ===")
    print(f"  Cases: {len(cases)}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {IMAGES_DIR}")
    print()
    
    # Prepare args for workers
    worker_args = [(s, c, seri, args.imgsz) for s, c, seri in cases]
    
    t_start = time.time()
    total_slices = 0
    done = 0
    failed = []
    
    with mp.Pool(args.workers, maxtasksperchild=10) as pool:
        for prefix, n, dt, err in pool.imap_unordered(worker, worker_args):
            done += 1
            total_slices += n
            if err:
                failed.append((prefix, err))
                print(f"  [{done}/{len(cases)}] {prefix}: FAILED ({err})")
            elif done % 20 == 0 or done <= 5:
                elapsed = time.time() - t_start
                rate = total_slices / elapsed if elapsed > 0 else 0
                remaining = (len(cases) - done) * (elapsed / done) if done > 0 else 0
                print(f"  [{done}/{len(cases)}] {prefix}: {n} slices ({dt:.1f}s) | "
                      f"total={total_slices} | {rate:.0f} sl/s | ETA={remaining/60:.0f}min")
    
    elapsed = time.time() - t_start
    print(f"\n=== Done ===")
    print(f"  Cases: {done}/{len(cases)}")
    print(f"  Slices: {total_slices}")
    print(f"  Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"  Rate: {total_slices/elapsed:.0f} slices/sec")
    print(f"  Failed: {len(failed)}")
    if failed:
        for f, e in failed[:10]:
            print(f"    {f}: {e}")
    
    # Disk usage
    total_size = 0
    for f in IMAGES_DIR.rglob("*.jpg"):
        total_size += f.stat().st_size
    print(f"  Disk: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
