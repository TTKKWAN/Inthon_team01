"""Make a small transfer bundle from a submission ZIP (see rebuild_zip.py).

Drops the three frozen encoder weight files (public on Hugging Face) and records, for each
of their tensors, shape / dtype / SHA-256 plus the HF revision whose float32 tensors cast
to float16 reproduce them bit for bit (checked here before writing the bundle).

  python make_bundle.py outputs/v2_debias_submission.zip --out C:/Users/ksjga/inthon2026_cache
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent
ENCODERS = {"dinov2": "facebook/dinov2-base",
            "mpnet": "sentence-transformers/all-mpnet-base-v2",
            "siglip": "google/siglip-base-patch16-224"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    name = args.zip.stem
    work = Path(tempfile.mkdtemp(prefix="bundle_")) / f"{name}_bundle"
    sub = work / "submission"
    with zipfile.ZipFile(args.zip) as z:
        files = [n for n in z.namelist() if not n.endswith("/")]
        z.extractall(sub)

    api, encoders = HfApi(), {}
    for enc, repo in ENCODERS.items():
        path = f"models/{enc}/model.safetensors"
        ours = load_file(sub / path)
        with safe_open(sub / path, "np") as f:
            metadata = f.metadata()
        rev = api.model_info(repo).sha
        hub = load_file(hf_hub_download(repo, "model.safetensors", revision=rev))
        tensors = {}
        for key, t in ours.items():
            if key not in hub or not np.array_equal(hub[key].astype(t.dtype), t):
                sys.exit(f"{enc}: {key} is not reproducible from {repo}@{rev}")
            tensors[key] = {"shape": list(t.shape), "dtype": str(t.dtype),
                            "sha256": hashlib.sha256(np.ascontiguousarray(t).tobytes())
                            .hexdigest()}
        encoders[enc] = {"repo": repo, "revision": rev, "path": path,
                         "metadata": metadata, "tensors": tensors}
        (sub / path).unlink()
        print(f"{enc}: {len(tensors)} tensors reproducible from {repo}@{rev[:10]}")

    (work / "encoder_manifest.json").write_text(json.dumps(
        {"zip_name": args.zip.name, "files": files, "encoders": encoders}), encoding="utf-8")
    shutil.copy(ROOT / "rebuild_zip.py", work / "rebuild_zip.py")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = shutil.make_archive(str(args.out / f"{name}_bundle"), "zip", work.parent, work.name)
    print(f"Bundle: {bundle} ({Path(bundle).stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
