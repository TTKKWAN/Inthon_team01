"""Shared CLI configuration and reproducibility helpers."""

import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def load_config(path=ROOT / "config.json"):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ["data_dir", "image_model_dir", "text_model_dir", "output_dir"]:
        config[key] = str((path.parent / config[key]).resolve())
    return config


def seed_everything(seed, cpu_threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(value="auto"):
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or a working GPU environment.")
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
