#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-$HOME/abdomen}
RUN_NAME=${RUN_NAME:-yolo11l-p2-1280}
source "$WORK/venv/bin/activate"
cd "$WORK"
mkdir -p logs pred
TS=$(date +%Y%m%d_%H%M%S)

echo "== smoke: 1 epoch, 1% train, full check of pipeline =="
python -u trainer/train.py \
  --data "$WORK/ds/dataset.yaml" \
  --model "$WORK/configs/yolo11l-p2.yaml" \
  --epochs 1 --fraction 0.01 --batch 24 --save-period -1 \
  --name smoke-1280 --exist-ok 2>&1 | tee "logs/smoke_${TS}.log"

echo "== full: 25 epochs, batch 24, imgsz 1280 =="
python -u trainer/train.py \
  --data "$WORK/ds/dataset.yaml" \
  --model "$WORK/configs/yolo11l-p2.yaml" \
  --epochs 25 --batch 24 --save-period 5 \
  --name "$RUN_NAME" --exist-ok 2>&1 | tee "logs/full_${TS}.log"

BEST="$WORK/runs/detect/abdomen/$RUN_NAME/weights/best.pt"

echo "== box-level mAP =="
python -u - <<PY
from ultralytics import YOLO
model = YOLO("$BEST")
for split in ("val", "test"):
    m = model.val(data="$WORK/ds/dataset.yaml", split=split, imgsz=1280, batch=24)
    print(split, "mAP50-95", round(float(m.box.map), 4), "mAP50", round(float(m.box.map50), 4))
PY

echo "== patient-level predictions =="
for split in val test; do
  python -u src/patient_eval.py predict --model "$BEST" \
    --images "$WORK/ds/images/$split" --out "$WORK/pred/$split.csv" \
    --imgsz 1280 --conf 0.005 --max-det 30 --batch 24 --device 0
done

echo "== patient-level metrics =="
GATES=""
if [ -f "$WORK/annotations/slice_gates.csv" ]; then
  GATES="--gates $WORK/annotations/slice_gates.csv"
elif [ -f "$WORK/dslabels/slice_gates.csv" ]; then
  GATES="--gates $WORK/dslabels/slice_gates.csv"
fi
python -u src/patient_eval.py eval \
  --val-pred "$WORK/pred/val.csv" --test-pred "$WORK/pred/test.csv" \
  --val-images "$WORK/ds/images/val" --test-images "$WORK/ds/images/test" \
  $GATES --score topk --topk 3 --grid --bootstrap 1000 \
  --out "$WORK/eval_patient"

echo "DONE: $WORK/eval_patient/report.json"
