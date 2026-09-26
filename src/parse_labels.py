"""
Parse Bilgi.xlsx (main) + Ek xlsx → unified annotations CSV.

Outputs:
  output/labels/annotations_bb.csv       — bounding box annotations (6 target classes)
  output/labels/annotations_boundary.csv — organ boundary slice annotations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BILGI_XLSX, EK_XLSX, LABELS_DIR,
    EN_CLASS_MAP, TR_CLASS_MAP,
    EN_ORGAN_MAP, TR_ORGAN_MAP,
    CLASS_NAMES,
)


def parse_bbox(val):
    """Parse 'x1,y1-x2,y2' → (x1, y1, x2, y2) floats, or None."""
    if pd.isna(val) or val == 0 or val == "0":
        return None
    s = str(val).strip()
    m = re.match(r"(\d+),(\d+)-(\d+),(\d+)", s)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return x1, y1, x2, y2
    return None


def parse_main():
    """Parse Bilgi.xlsx → list of BB rows + boundary rows."""
    bb_rows = []
    bnd_rows = []

    xl = pd.ExcelFile(BILGI_XLSX)
    for sheet in xl.sheet_names:
        source = "egitim" if "TRAII" in sheet.upper() else "yarisma"
        df = pd.read_excel(xl, sheet_name=sheet)

        for _, row in df.iterrows():
            case = int(row["Case Number"])
            img_id = str(row["Image Id"])
            typ = row["Type"]
            cls = str(row["Class"]).strip()
            data = row["Data"]

            if typ == "Bounding Box":
                # Check if it's a pathology or organ
                if cls in EN_CLASS_MAP:
                    cid = EN_CLASS_MAP[cls]
                    bbox = parse_bbox(data)
                    if bbox is not None:
                        bb_rows.append({
                            "source": source,
                            "case": case,
                            "image_id": img_id,
                            "class_id": cid,
                            "class_name": CLASS_NAMES[cid],
                            "x1": bbox[0], "y1": bbox[1],
                            "x2": bbox[2], "y2": bbox[3],
                        })
                elif cls in EN_ORGAN_MAP:
                    # Organ BB (rare, only in training) — skip for detection
                    pass
                else:
                    print(f"  [WARN] Unknown BB class: '{cls}' (sheet={sheet}, case={case})")

            elif typ == "Boundary Slice":
                if cls in EN_ORGAN_MAP:
                    organ = EN_ORGAN_MAP[cls]
                    bnd_rows.append({
                        "source": source,
                        "case": case,
                        "image_id": img_id,
                        "organ": organ,
                        "data": 0 if pd.isna(data) else int(data),
                    })
                elif cls in EN_CLASS_MAP:
                    # Some pathologies appear as boundary slice too
                    # (marks start/end of pathology visibility)
                    cid = EN_CLASS_MAP[cls]
                    bnd_rows.append({
                        "source": source,
                        "case": case,
                        "image_id": img_id,
                        "organ": f"path_{CLASS_NAMES[cid]}",
                        "data": 0 if pd.isna(data) else int(data),
                    })
                else:
                    print(f"  [WARN] Unknown boundary class: '{cls}' (sheet={sheet}, case={case})")

    return bb_rows, bnd_rows


def parse_ek():
    """Parse Ek xlsx → list of BB rows + boundary rows."""
    bb_rows = []
    bnd_rows = []

    df = pd.read_excel(EK_XLSX)

    for _, row in df.iterrows():
        case_str = str(row["Olgu Numarası"])       # e.g. "10238/Seri4"
        kesit = str(row["Kesit Numarası"])          # SOP Instance UID (11-digit)
        tip = row["Tip"]
        sinif = str(row["Sınıf"]).strip()
        veri = row["Veri"]

        # Parse case number from "10238/Seri4"
        m = re.match(r"(\d+)/Seri(\d+)", case_str)
        if m:
            case = int(m.group(1))
            seri = int(m.group(2))
        else:
            print(f"  [WARN] Unparseable Ek case: '{case_str}'")
            continue

        if tip == "BB":
            if sinif in TR_CLASS_MAP:
                cid = TR_CLASS_MAP[sinif]
                bbox = parse_bbox(veri)
                if bbox is not None:
                    bb_rows.append({
                        "source": "ek",
                        "case": case,
                        "seri": seri,
                        "image_id": kesit,  # SOP Instance UID
                        "class_id": cid,
                        "class_name": CLASS_NAMES[cid],
                        "x1": bbox[0], "y1": bbox[1],
                        "x2": bbox[2], "y2": bbox[3],
                    })
            elif sinif in TR_ORGAN_MAP:
                pass  # organ BB, skip for detection
            else:
                print(f"  [WARN] Unknown Ek BB class: '{sinif}' (case={case_str})")

        elif tip in ("Başlangıç Kesiti", "Bitiş Kesiti"):
            is_start = (tip == "Başlangıç Kesiti")
            if sinif in TR_ORGAN_MAP:
                organ = TR_ORGAN_MAP[sinif]
                bnd_rows.append({
                    "source": "ek",
                    "case": case,
                    "seri": seri,
                    "image_id": kesit,
                    "organ": organ,
                    "boundary_type": "start" if is_start else "end",
                    "data": 0 if pd.isna(veri) else int(veri),
                })
            elif sinif in TR_CLASS_MAP:
                cid = TR_CLASS_MAP[sinif]
                bnd_rows.append({
                    "source": "ek",
                    "case": case,
                    "seri": seri,
                    "image_id": kesit,
                    "organ": f"path_{CLASS_NAMES[cid]}",
                    "boundary_type": "start" if is_start else "end",
                    "data": 0 if pd.isna(veri) else int(veri),
                })
            else:
                print(f"  [WARN] Unknown Ek boundary class: '{sinif}' (case={case_str})")

    return bb_rows, bnd_rows


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Parsing Bilgi.xlsx (main) ===")
    main_bb, main_bnd = parse_main()
    print(f"  BB rows: {len(main_bb)}, Boundary rows: {len(main_bnd)}")

    print("\n=== Parsing Ek xlsx ===")
    ek_bb, ek_bnd = parse_ek()
    print(f"  BB rows: {len(ek_bb)}, Boundary rows: {len(ek_bnd)}")

    # Combine
    all_bb = main_bb + ek_bb
    all_bnd = main_bnd + ek_bnd

    bb_df = pd.DataFrame(all_bb)
    bnd_df = pd.DataFrame(all_bnd)

    # Stats
    print(f"\n=== Total BB annotations: {len(bb_df)} ===")
    print(bb_df.groupby("class_name")["case"].nunique().to_string())
    print(f"\n  Total unique patients: {bb_df['case'].nunique()}")
    print(f"  Sources: {bb_df.groupby('source')['case'].nunique().to_dict()}")

    print(f"\n=== Total boundary annotations: {len(bnd_df)} ===")
    print(bnd_df.groupby("organ")["case"].nunique().to_string())

    # Save
    bb_path = LABELS_DIR / "annotations_bb.csv"
    bnd_path = LABELS_DIR / "annotations_boundary.csv"
    bb_df.to_csv(bb_path, index=False)
    bnd_df.to_csv(bnd_path, index=False)
    print(f"\nSaved: {bb_path} ({len(bb_df)} rows)")
    print(f"Saved: {bnd_path} ({len(bnd_df)} rows)")


if __name__ == "__main__":
    main()
