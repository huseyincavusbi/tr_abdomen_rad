"""Inject credentials into the staged notebook and push it to Kaggle.

Usage:
  WANDB_API_KEY=... python trainer/push_staged.py
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

SRC = Path("notebooks/staged")
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def escape(value):
    return json.dumps(value)[1:-1]


def main():
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise SystemExit("set WANDB_API_KEY")
    kaggle_json = KAGGLE_JSON.read_text().strip()

    tmp = Path(tempfile.mkdtemp(prefix="staged_push_"))
    shutil.copytree(SRC, tmp, dirs_exist_ok=True)
    nb = tmp / "kaggle_train_staged.ipynb"
    text = nb.read_text()
    text = text.replace("__KAGGLE_JSON__", escape(kaggle_json))
    text = text.replace("__WANDB_API_KEY__", escape(wandb_key))
    nb.write_text(text)

    api = KaggleApi()
    api.authenticate()
    api.kernels_push(str(tmp))
    print("pushed:", tmp, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
