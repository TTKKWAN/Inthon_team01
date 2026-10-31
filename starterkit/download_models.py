"""Download only the two allowed baseline encoders, or copy existing local snapshots."""

import argparse
import shutil
from pathlib import Path

from baseline_model import IMAGE_MODEL_ID, TEXT_MODEL_ID
from common import ROOT, load_config, save_json
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source-root", type=Path,
                        help="Optional local directory containing dinov2/ and mpnet/ snapshots.")
    args = parser.parse_args()
    config = load_config(args.config)
    for name, model_id, key in [
        ("dinov2", IMAGE_MODEL_ID, "image_model_dir"),
        ("mpnet", TEXT_MODEL_ID, "text_model_dir"),
    ]:
        target = Path(config[key])
        if args.source_root:
            source = (args.source_root / name).resolve()
            if source == target:
                raise ValueError("Source and destination model directories must differ.")
            if not (source / "config.json").is_file() or not list(source.glob("*.safetensors")):
                raise FileNotFoundError(f"Local model snapshot is incomplete: {source}")
            shutil.copytree(source, target, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".cache", "__pycache__"))
        else:
            snapshot_download(
                model_id, local_dir=target,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
            )
        save_json(target / "baseline_source.json", {"model_id": model_id})
        print(f"Ready: {model_id} -> {target}", flush=True)


if __name__ == "__main__":
    main()
