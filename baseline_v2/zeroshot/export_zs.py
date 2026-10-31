"""Package the SigLIP zero-shot control submission (no trained weights).

  python zeroshot/export_zs.py --siglip models/siglip --out ../submission/siglip_zs

Writes <out>/ (submission.py, submission.json, siglip/ in fp16 with tokenizer.json) and
<out>.zip with submission.py at the ZIP root.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--siglip", type=Path, required=True, help="google/siglip-base-patch16-224 dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty directory: {out}")
    model = AutoModel.from_pretrained(args.siglip, local_files_only=True, dtype=torch.float32)
    if model.config.model_type != "siglip":
        raise ValueError("Expected google/siglip-base-patch16-224")
    (out / "siglip").mkdir(parents=True, exist_ok=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    model.save_pretrained(out / "siglip", state_dict=state, safe_serialization=True)
    AutoImageProcessor.from_pretrained(args.siglip, local_files_only=True,
                                       use_fast=False).save_pretrained(out / "siglip")
    shutil.copyfile(args.siglip / "tokenizer.json", out / "siglip/tokenizer.json")
    shutil.copyfile(HERE / "submission.py", out / "submission.py")
    (out / "submission.json").write_text(json.dumps({"api_version": 2}), encoding="utf-8")
    (out / "model_source.json").write_text(json.dumps(
        {"model_id": "google/siglip-base-patch16-224", "training": "none (zero-shot)"}),
        encoding="utf-8")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(out).as_posix())
    files = len(zipfile.ZipFile(zip_path).namelist())
    print(f"Exported {out} and {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MiB, {files} files)")


if __name__ == "__main__":
    main()
