"""Build a self-contained submission folder + ZIP from a train_v2 checkpoint."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch
from model_v2 import FrozenEncoders

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission_v2")
    parser.add_argument("--int8", action="store_true",
                        help="store encoder weights as per-row int8 (~half the ZIP size)")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty directory: {out}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = FrozenEncoders(ROOT / cfg["image_model_dir"], ROOT / cfg["text_model_dir"],
                         ROOT / cfg["siglip_model_dir"], "cpu")
    out.mkdir(parents=True, exist_ok=True)
    enc.export(out / "models", int8=args.int8)
    (out / "weights").mkdir()
    torch.save(ckpt["state_dict"], out / "weights/scorer.pt")
    for name in ["submission.py", "model_v2.py"]:
        shutil.copyfile(ROOT / name, out / name)
    (out / "submission.json").write_text(json.dumps({"api_version": 2}), encoding="utf-8")
    (out / "inference_config.json").write_text(json.dumps(
        {"model_config": ckpt["model_config"], "cpu_threads": 4}, indent=2), encoding="utf-8")
    (out / "training_info.json").write_text(json.dumps(
        {"epoch": ckpt["epoch"], "metrics": ckpt["metrics"], "config": cfg}, indent=2),
        encoding="utf-8")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(out).as_posix())
    print(f"Exported {out} and {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
