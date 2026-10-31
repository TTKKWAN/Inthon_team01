"""Package an exported submission; ZIP root contains submission.py and its local models."""

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

from common import ROOT

REQUIRED = {
    "submission.py", "baseline_model.py", "submission.json", "inference_config.json",
    "training_info.json", "weights/mlp.pt", "models/dinov2/config.json",
    "models/dinov2/model.safetensors", "models/dinov2/preprocessor_config.json",
    "models/mpnet/config.json", "models/mpnet/model.safetensors",
    "models/mpnet/tokenizer_config.json", "models/mpnet/tokenizer.json",
}


def package(source, output, force=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and not force:
        raise FileExistsError(f"ZIP already exists; choose a new --output or pass --force: {output}")
    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Do not submit symbolic links: {relative}")
        if path.is_file() and (relative.as_posix() in REQUIRED
                               or relative.parts[0] == "models"):
            files.append(relative)
    missing = REQUIRED - {path.as_posix() for path in files}
    if missing:
        raise ValueError(f"Incomplete export: {sorted(missing)}")
    expanded_size = sum((source / path).stat().st_size for path in files)
    if expanded_size > 1024 ** 3 or len(files) > 10000:
        raise ValueError("Server contract: expanded size <= 1 GiB and files <= 10,000.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for relative in files:
                archive.write(source / relative, relative.as_posix())
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Created: {output}; ZIP={output.stat().st_size / 2**20:.1f} MiB; "
          f"expanded={expanded_size / 2**20:.1f} MiB", flush=True)
    if output.stat().st_size > 256 * 2**20:
        print("Operator setting required: server MAX_UPLOAD_BYTES must exceed this ZIP's size "
              "(the current server default is 256 MiB).", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "sample_submission")
    parser.add_argument("--output", type=Path, default=ROOT / "sample_submission.zip")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    package(args.submission_dir, args.output, args.force)


if __name__ == "__main__":
    main()
