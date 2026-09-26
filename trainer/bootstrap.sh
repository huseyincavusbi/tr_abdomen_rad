#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-$HOME/abdomen}
MIN_DISK_GB=${MIN_DISK_GB:-150}
mkdir -p "$WORK"
cd "$WORK"

echo "== [1/7] system packages =="
missing=""
command -v unzip >/dev/null 2>&1 || missing="$missing unzip"
command -v pip3 >/dev/null 2>&1 || missing="$missing python3-pip"
python3 -c "import venv" >/dev/null 2>&1 || missing="$missing python3-venv"
ldconfig -p 2>/dev/null | grep -q "libGL.so.1" || missing="$missing libgl1 libglib2.0-0"
if [ -n "$missing" ]; then
  echo "installing:$missing"
  apt-get -qq update
  apt-get -qq install -y $missing
else
  echo "system packages OK"
fi

echo "== [2/7] disk =="
avail_gb=$(df -Pk "$WORK" | awk 'NR==2{print int($4/1024/1024)}')
echo "free: ${avail_gb} GB at $WORK"
if [ "$avail_gb" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: need >=${MIN_DISK_GB} GB free (override with MIN_DISK_GB)"
  exit 1
fi

echo "== [3/7] python env =="
if [ ! -d "$WORK/venv" ]; then python3 -m venv "$WORK/venv"; fi
source "$WORK/venv/bin/activate"
pip install -q -U pip wheel
pip install -q "ultralytics==8.4.93" kaggle scikit-learn
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
      "vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else 0)
PY
if [ ! -f "$WORK/src/patient_eval.py" ]; then
  echo "WARNING: $WORK/src/patient_eval.py missing (needed for patient-level eval)"
fi

echo "== [4/7] kaggle credentials =="
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  echo "ERROR: place kaggle.json in ~/.kaggle/ or export KAGGLE_USERNAME + KAGGLE_KEY"
  exit 1
fi
chmod 600 "$HOME/.kaggle/kaggle.json" 2>/dev/null || true

echo "== [5/7] download images (43 GB) =="
if [ ! -d "$WORK/dl/yolo_dataset/images/train" ]; then
  mkdir -p "$WORK/dl"
  kaggle datasets download -d huseyincavus/abdomen-prepared -p "$WORK/dl" --unzip -q
  if [ -z "${KEEP_ARCHIVES:-}" ]; then rm -f "$WORK/dl"/*.zip; fi
else
  echo "already present, skipping"
fi

echo "== [6/7] download labels =="
if [ ! -d "$WORK/dslabels/labels/train" ]; then
  mkdir -p "$WORK/dslabels"
  kaggle datasets download -d huseyincavus/abdomen-yolo-labels -p "$WORK/dslabels" --unzip -q
  if [ -z "${KEEP_ARCHIVES:-}" ]; then rm -f "$WORK/dslabels"/*.zip; fi
else
  echo "already present, skipping"
fi
if [ ! -d "$WORK/dslabels/labels/train" ] && [ -f "$WORK/dslabels/labels.zip" ]; then
  ( cd "$WORK/dslabels" && unzip -q -o labels.zip )
fi
if [ ! -d "$WORK/dslabels/labels/train" ]; then
  echo "ERROR: labels not found after download"
  exit 1
fi
if [ ! -f "$WORK/annotations/slice_gates.csv" ] && [ ! -f "$WORK/dslabels/slice_gates.csv" ]; then
  echo "WARNING: slice_gates.csv missing (gating will be disabled)"
fi
python "$WORK/dslabels/assemble_dataset.py" \
  --images "$WORK/dl/yolo_dataset/images" \
  --labels "$WORK/dslabels/labels" \
  --out "$WORK/ds"

echo "== [7/7] sanity =="
python "$WORK/trainer/sanity_check.py" \
  --images "$WORK/ds/images" \
  --labels "$WORK/ds/labels" \
  --out "$WORK/ds/sanity_report.json" \
  --montage "$WORK/ds/montage" \
  --image-size 1280

echo "READY: $WORK/ds/dataset.yaml"
