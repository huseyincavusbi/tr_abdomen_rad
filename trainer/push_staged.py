"""Materialize the staged notebook with credentials outside git and push it.

Usage:
  WANDB_API_KEY=... python trainer/push_staged.py
  python trainer/push_staged.py --no-push
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

SRC = Path("notebooks/staged")
OUT = Path("output/staged_live")
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"
WANDB_KEY_FILE = Path("output/.wandb_key")


def escape(value):
    return json.dumps(value)[1:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key and WANDB_KEY_FILE.exists():
        wandb_key = WANDB_KEY_FILE.read_text().strip()
    if not wandb_key:
        raise SystemExit("set WANDB_API_KEY or write output/.wandb_key")

    kaggle_json = KAGGLE_JSON.read_text().strip()

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    nb = OUT / "kaggle_train_staged.ipynb"
    text = nb.read_text()
    text = text.replace("__KAGGLE_JSON__", escape(kaggle_json))
    text = text.replace("__WANDB_API_KEY__", escape(wandb_key))
    nb.write_text(text)
    print("live copy:", OUT, flush=True)

    if not args.no_push:
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(OUT))
        print("pushed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
