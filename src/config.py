"""
Central configuration: class mappings, paths, HU windows, training constants.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "tr-abdomen-rad"
EGITIM = DATA / "egitim" / "extracted"
YARISMA = DATA / "yarisma" / "extracted" / "Yarìƒma Veri Seti"
EK = DATA / "ek" / "extracted"
BILGI_XLSX = DATA / "Bilgi.xlsx"
EK_XLSX = DATA / "ek" / "Training set-extra(1-5).xlsx"
ANNOTATIONS_DIR = ROOT / "annotations"
OUTPUT = ROOT / "output"
LABELS_DIR = OUTPUT / "labels"
IMAGES_DIR = OUTPUT / "images"
SPLITS_DIR = OUTPUT / "splits"

# ─── 6 target pathology classes (YOLO class IDs 0-5) ─────────────────────
# Per 2023 paper Table: competitor expected output labels
#   1-Acute appendicitis, 2-Acute cholecystitis, 3-Acute pancreatitis,
#   4-Kidney/ureter stone, 5-Acute diverticulitis, 6-AAA/dissection
# * items are "additional information" sub-findings mapped to parent class

CLASS_NAMES = [
    "acute_appendicitis",
    "acute_cholecystitis",
    "acute_pancreatitis",
    "kidney_ureteral_stone",
    "acute_diverticulitis",
    "aaa_dissection",
]
NUM_CLASSES = len(CLASS_NAMES)

# English class name → YOLO class ID
EN_CLASS_MAP = {
    # Class 0: acute appendicitis
    "Compatible with acute appendicitis": 0,
    "Appendicolith": 0,
    # Class 1: acute cholecystitis
    "Compatible with acute cholecystitis": 1,
    "Gallbladder stone": 1,
    "Gall bladder stone": 1,
    # Class 2: acute pancreatitis
    "Compatible with acute pancreatitis": 2,
    # Class 3: kidney/ureteral stone
    "Kidney stone": 3,
    "ureteral stone": 3,
    "Ureter stone": 3,
    # Class 4: acute diverticulitis
    "Compatible with acute diverticulitis": 4,
    "Calcified diverticulum": 4,
    # Class 5: AAA/dissection
    "Abdominal aortic aneurysm": 5,
    "Abdominal aortic dissection": 5,
}

# Turkish class name → YOLO class ID (Ek dataset)
TR_CLASS_MAP = {
    # Class 0
    "Akut apandisit ile uyumlu": 0,
    "Apendikolit": 0,
    # Class 1
    "Akut kolesistit ile uyumlu": 1,
    "Safra kesesi taşı": 1,
    # Class 2
    "Akut pankreatit ile uyumlu": 2,
    # Class 3
    "Böbrek taşı": 3,
    "Üreter taşı": 3,
    # Class 4
    "Akut divertikülit ile uyumlu": 4,
    "Kalsifiye divertikül": 4,
    # Class 5
    "Abdominal aort anevrizma": 5,
}

# Organ landmarks (boundary slices, NOT detection targets)
# English → canonical organ name
EN_ORGAN_MAP = {
    "Abdominal Aorta": "abdominal_aorta",
    "Pancreas": "pancreas",
    "Gall bladder": "gall_bladder",
    "Kidney-Bladder": "kidney_bladder",
    "Colon": "colon",
    "appendix": "appendix",
}

TR_ORGAN_MAP = {
    "Abdominal Aorta": "abdominal_aorta",
    "Pankreas": "pancreas",
    "Safra Kesesi": "gall_bladder",
    "Böbrek-Mesane": "kidney_bladder",
    "Kolon": "colon",
    "Apandiks": "appendix",
}

ORGAN_NAMES = list(EN_ORGAN_MAP.values())

# ─── HU windows (per 2026 paper) ─────────────────────────────────────────
# Each window: (center L, width W) → maps to 0-255 channel
# R: soft tissue  L50/W400
# G: bone/stone   L500/W1600
# B: angio/liver  L60/W150
HU_WINDOWS = [
    (50, 400),      # R: soft tissue
    (500, 1600),    # G: bone/stone
    (60, 150),      # B: angio/liver
]

# ─── Training constants ──────────────────────────────────────────────────
IMGSZ = 1280
NEG_POS_RATIO = 3          # 3:1 negative:positive slice ratio
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train/val/test (patient-level)
RANDOM_SEED = 42

# Anatomic envelope: ±40mm around organ boundary slices
ENVELOPE_MM = 40
