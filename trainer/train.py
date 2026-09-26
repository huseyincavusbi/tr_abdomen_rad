"""Train YOLO11-L-P2 @1280 per Erdogan & Koc 2026 on the assembled dataset.

Usage:
  python trainer/train.py --data ds/dataset.yaml --model configs/yolo11l-p2.yaml --weights yolo11l.pt
  python trainer/train.py --data ds/dataset.yaml --resume runs/detect/abdomen/yolo11l-p2-1280/weights/last.pt
"""
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="configs/yolo11l-p2.yaml")
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--project", default="abdomen")
    ap.add_argument("--name", default="yolo11l-p2-1280")
    ap.add_argument("--lr0", type=float, default=0.0007)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--exist-ok", action="store_true")
    return ap.parse_args()


def patch_amp(dtype_name):
    import torch
    import ultralytics.engine.trainer as trainer_mod
    import ultralytics.engine.validator as validator_mod

    if dtype_name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 requested but not supported on this GPU; use --amp-dtype fp16")
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    def autocast(enabled, device="cuda"):
        return torch.amp.autocast(device, enabled=enabled, dtype=dtype)

    trainer_mod.autocast = autocast
    validator_mod.autocast = autocast
    return dtype


def main():
    args = parse_args()
    import torch
    from ultralytics import YOLO

    dtype = patch_amp(args.amp_dtype)
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} "
        f"amp={args.amp_dtype} amp_dtype={dtype}"
    )

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    if args.amp_dtype == "bf16":
        def disable_scaler(trainer):
            trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)

        model.add_callback("on_train_start", disable_scaler)

    model.train(
        data=args.data,
        pretrained=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        amp=True,
        mosaic=1.0,
        close_mosaic=args.close_mosaic,
        fliplr=0.0,
        patience=args.patience,
        save_period=args.save_period,
        fraction=args.fraction,
        seed=args.seed,
        cache=False,
        rect=False,
        val=True,
        plots=True,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
