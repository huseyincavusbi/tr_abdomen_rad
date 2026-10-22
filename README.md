# TR-ABDOMEN-RAD — Acute Abdominal Emergency Detection

Six-class acute abdominal emergency detection on CT with patient-level evaluation —
built independently for the TEKNOFEST 2026 TR_ABDOMEN_RAD task.

## Data

| Split | Studies | Images | Positive slices | Boxes |
| :--- | ---: | ---: | ---: | ---: |
| TR_ABDOMEN_RAD (907 / 194 / 195) | 1,296 | 468,836 | 39,614 | 43,166 |

Classes: acute appendicitis, acute cholecystitis, acute pancreatitis,
kidney/ureteral stone, acute diverticulitis, AAA/dissection.

Labels were rebuilt from the official annotations with source-aware joins. Fixed
dataset issues: ×1280 coordinate scaling, 258 foreign labels, 5 degenerate boxes.
The demo run uses a balanced, patient-disjoint subset — 10,764 train / 3,148 val / 2,828 test images. Training covered 6.8% of the positive slices (2.3% of all slices) and 6.8% of boxes; the full dataset was not used.

## Results

| Metric | Val (36 patients) | Test (26 patients) |
| :--- | ---: | ---: |
| Macro AUROC | 0.879 [0.808–0.935] | **0.921** [0.867–0.967] |
| Macro F1 | 0.781 | 0.664 |
| Precision | 0.763 | 0.644 |
| Recall | 0.806 | 0.733 |
| Localization (restricted) | 0.889 | 0.870 |
| Localization (coverage-incl.) | 0.711 | 0.645 |
| Detection mAP50 (slice) | 0.220 | — |

YOLO26-S-P2 · 9.67 M params · 1280 px · 10 epochs · bf16 on one 16 GB GPU.
Bootstrap CIs over 1,000 patient resamples. Weakest class: diverticulitis
(AUROC 0.692); strongest: AAA (1.000).

*Demo-scale run, not a paper reproduction: 10 epochs on 6.8% of the positive
slices (2.3% of all slices), 26 test patients. Specificity is not evaluated.*

## Approach

- DICOM → three HU windows (soft tissue / bone / angio) as an RGB image.
- YOLO26-S-P2 detection at 1280 px, bf16, AdamW, mosaic for the first 6/10 epochs.
- Balanced, patient-disjoint subset — no patient appears in two splits.
- Patient score = top-k mean over anatomically gated slices; thresholds tuned on val.
- Anatomical slice gates restrict each class to plausible slice ranges (0.06% of GT removed).

## Findings

- Patient-level ranking is much easier than localization: AUROC 0.92 vs mAP50 0.22.
- Closing mosaic for the last 4 epochs gave the biggest jump (mAP50 0.165 → 0.222).
- Diverticulitis is the weak class (AUROC 0.692) — few examples, subtle presentation.
- The 10-epoch curve plateaued; the next levers are data fraction and model size.

## Quickstart

```bash
pip install ultralytics==8.4.93 scikit-learn pandas

# 1. raw DICOMs -> 3-window RGB + corrected labels
python src/convert_all.py
python src/build_dataset.py

# 2. balanced, patient-disjoint demo subset
python src/make_balanced_subset.py

# 3. train (bf16)
python trainer/train.py --data output/subset/dataset.yaml \
  --model configs/yolo26s-p2.yaml --weights yolo26s.pt \
  --epochs 10 --batch 4 --imgsz 1280 --amp-dtype bf16 --device 0

# 4. patient-level evaluation
python src/patient_eval.py predict --model runs/detect/abdomen/demo/weights/best.pt \
  --images output/subset/images/test --out pred/test.csv \
  --imgsz 1280 --conf 0.005 --batch 8

python src/patient_eval.py eval --val-pred pred/val.csv --test-pred pred/test.csv \
  --val-images output/subset/images/val --test-images output/subset/images/test \
  --gates annotations/slice_gates.csv --score topk --topk 3 --grid --bootstrap 1000 \
  --out eval
```

## Layout

```
src/          DICOM conversion, dataset build/subset, slice gates, patient-level eval
trainer/      bf16 training entry point, sanity check, bootstrap
configs/      YOLO26 / YOLO11 P2 detector configs
annotations/  ground-truth exports and anatomical slice gates
```

## Weights & artifacts

- [tr-abdomen-rad-weights](https://huggingface.co/huseyincavus/tr-abdomen-rad-weights) —
  demo model, training checkpoints, evaluation report, predictions, plots and demo overlays.

## References

- Erdoğan & Koç (2026), TR_ABDOMEN_RAD baseline.
- TEKNOFEST 2026 TR_ABDOMEN_RAD dataset.
