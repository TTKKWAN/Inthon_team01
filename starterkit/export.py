"""Create a self-contained server API v2 submission from a trained MLP checkpoint."""

import argparse
import shutil
from pathlib import Path

import torch
from baseline_model import FrozenEncoders
from common import ROOT, save_json


def export(checkpoint_path, output_dir):
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use an empty submission directory: {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported baseline checkpoint format.")
    config = checkpoint["config"]
    torch.set_num_threads(config["cpu_threads"])
    encoders = FrozenEncoders(config["image_model_dir"], config["text_model_dir"], "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoders.export(output_dir / "models")
    (output_dir / "weights").mkdir(exist_ok=True)
    torch.save(checkpoint["state_dict"], output_dir / "weights/mlp.pt")
    for name in ["submission.py", "baseline_model.py"]:
        shutil.copyfile(ROOT / name, output_dir / name)
    save_json(output_dir / "submission.json", {"api_version": 2})
    save_json(output_dir / "inference_config.json", {
        "model_config": checkpoint["model_config"],
        "image_batch_size": config["image_batch_size"],
        "text_batch_size": config["text_batch_size"],
        "max_text_length": config["max_text_length"], "cpu_threads": config["cpu_threads"],
    })
    save_json(output_dir / "training_info.json", {
        **checkpoint["provenance"], "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"], "encoder_storage_dtype": "float16",
        "note": "Encoder weights are stored in float16; CPU inference casts them to float32.",
    })
    for name, model_id in [("dinov2", checkpoint["provenance"]["image_model_id"]),
                           ("mpnet", checkpoint["provenance"]["text_model_id"])]:
        save_json(output_dir / "models" / name / "baseline_source.json", {"model_id": model_id})
    print(f"Exported self-contained submission: {output_dir.resolve()}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sample_submission")
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
