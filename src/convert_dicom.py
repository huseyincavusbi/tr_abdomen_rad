"""
DICOM → multi-window HU-encoded PNG converter.

Per 2026 paper: 3-channel RGB encoding
  R = soft tissue  (L50, W400)
  G = bone/stone   (L500, W1600)
  B = angio/liver  (L60, W150)
"""

import pydicom
import numpy as np
import cv2
from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from config import HU_WINDOWS, EGITIM, YARISMA, EK, IMAGES_DIR


def apply_hu_window(hu_array, center, width):
    """Apply HU window → 0-255 uint8."""
    lo = center - width / 2.0
    hi = center + width / 2.0
    arr = np.clip(hu_array, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def dicom_to_hu(ds):
    """Read DICOM → HU array (float32). Handles multi-frame/excess data."""
    pixel = ds.pixel_array
    # Handle excess frames (some files have 2 frames but NumberOfFrames=1)
    if pixel.ndim == 3 and pixel.shape[0] in (2, 3, 4):
        pixel = pixel[0]
    pixel = pixel.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixel * slope + intercept
    return hu


def dicom_to_3window_png(ds, target_size=None):
    """
    DICOM dataset → 3-window HU-encoded RGB uint8 array.
    If target_size given, resize to (target_size, target_size).
    """
    hu = dicom_to_hu(ds)
    h, w = hu.shape[:2]

    channels = []
    for center, width in HU_WINDOWS:
        channels.append(apply_hu_window(hu, center, width))
    rgb = np.stack(channels, axis=-1)  # H, W, 3

    if target_size is not None and (h, w) != (target_size, target_size):
        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return rgb


def convert_one(dcm_path, out_path, target_size=None, jpeg_quality=90):
    """Convert single DICOM file → image (JPEG or PNG based on extension)."""
    try:
        ds = pydicom.dcmread(str(dcm_path))
        rgb = dicom_to_3window_png(ds, target_size)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if str(out_path).endswith(".jpg"):
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(str(out_path), bgr)
        return ds.Rows, ds.Columns
    except Exception as e:
        print(f"  [ERROR] {dcm_path}: {e}")
        return None


def find_case_dir(source, case_num, seri=None):
    """Find the directory containing DICOMs for a case."""
    if source == "egitim":
        d = EGITIM / str(case_num)
        if d.exists():
            return d
    elif source == "yarisma":
        d = YARISMA / str(case_num)
        if d.exists():
            return d
    elif source == "ek":
        # Ek directory structure (after extraction):
        #   EK / "Ek eğitim seti-UH1" / {case} / Seri{M} / *.dcm   (UH1, no prefix)
        #   EK / UH2 / {case} / Seri{M} / *.dcm                    (UH2-5, with prefix)
        if not EK.exists():
            return None
        seri_dir = f"Seri{seri}" if seri is not None else None
        for uh_dir in sorted(EK.iterdir()):
            if not uh_dir.is_dir():
                continue
            # Direct: uh_dir/case/Seri{M}
            if seri_dir:
                case_path = uh_dir / str(case_num) / seri_dir
                if case_path.exists():
                    return case_path
                # One level deeper: uh_dir/UHx_prefix/case/Seri{M}
                for sub in uh_dir.iterdir():
                    if sub.is_dir():
                        case_path = sub / str(case_num) / seri_dir
                        if case_path.exists():
                            return case_path
            else:
                case_path = uh_dir / str(case_num)
                if case_path.exists():
                    return case_path
                for sub in uh_dir.iterdir():
                    if sub.is_dir():
                        case_path = sub / str(case_num)
                        if case_path.exists():
                            return case_path
    return None


def list_dicoms(case_dir):
    """List DICOM files in anatomical order (sorted by filename = InstanceNumber order)."""
    files = sorted([f for f in os.listdir(case_dir) if f.endswith(".dcm")])
    return [case_dir / f for f in files]


def native_dims_of(dcm_path):
    """Read native (Rows, Columns) from DICOM header without pixel data."""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return int(ds.Rows), int(ds.Columns)
    except Exception:
        return None


def convert_case(source, case_num, seri=None, target_size=None, overwrite=False, fmt="jpg"):
    """
    Convert all DICOMs in a case → images.
    fmt: "jpg" (JPEG quality 90) or "png"
    Returns dict: {image_id: (native_h, native_w)} for successful conversions.
    Native dims are always returned (independent of target_size) so that
    annotations, which live in native pixel space, can be normalized correctly.
    image_id = filename without .dcm extension
    """
    case_dir = find_case_dir(source, case_num, seri)
    if case_dir is None or not case_dir.exists():
        return None

    # Output dir: images/{source}_{case}/
    prefix = f"{source}_{case_num}" if seri is None else f"{source}_{case_num}_s{seri}"
    out_dir = IMAGES_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = list_dicoms(case_dir)
    results = {}
    ext = "jpg" if fmt == "jpg" else "png"

    for dcm_path in dcm_files:
        img_id = dcm_path.stem  # filename without .dcm
        out_path = out_dir / f"{img_id}.{ext}"
        if out_path.exists() and not overwrite:
            dims = native_dims_of(dcm_path)
            if dims is not None:
                results[img_id] = dims
            continue
        dims = convert_one(dcm_path, out_path, target_size)
        if dims is not None:
            results[img_id] = dims

    return results


def build_sopuid_lookup(case_dir):
    """
    Build SOPInstanceUID → filename mapping for a case directory.
    Needed for Ek dataset where slice IDs are SOPInstanceUIDs.
    Returns dict: {sop_uid_short: filename}
    """
    mapping = {}
    dcm_files = list_dicoms(case_dir)
    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            uid = ds.SOPInstanceUID
            mapping[uid] = dcm_path.stem
            # Also map last 11 digits (Ek kesit numarası format)
            digits = "".join(c for c in uid if c.isdigit())
            if len(digits) >= 11:
                mapping[digits[-11:]] = dcm_path.stem
        except Exception:
            continue
    return mapping


if __name__ == "__main__":
    # Quick test: convert case 20001
    print("=== Test: convert case 20001 (egitim) ===")
    results = convert_case("egitim", 20001, target_size=640)
    if results:
        print(f"  Converted {len(results)} slices")
        for img_id, dims in list(results.items())[:3]:
            print(f"    {img_id}: {dims}")
        print(f"  Output: {IMAGES_DIR / 'egitim_20001'}")
    else:
        print("  FAILED")
