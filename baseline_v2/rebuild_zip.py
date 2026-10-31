"""Rebuild a full submission ZIP from a small bundle + the public encoders on Hugging Face.

The bundle holds everything in the ZIP except the three frozen encoder weight files
(DINOv2, MPNet, SigLIP: ~700 MB, public). This script downloads them at the pinned
revisions, casts float tensors to float16 exactly as the export did, checks EVERY tensor
against the SHA-256 recorded from the original ZIP, and only then writes the ZIP.

Needs only: pip install numpy safetensors huggingface_hub   (no torch)

  python rebuild_zip.py                      # bundle = this folder, ZIP next to it
  python rebuild_zip.py --out ~/Downloads/v2_debias_submission.zip
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=HERE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "encoder_manifest.json").read_text(encoding="utf-8"))
    sub = args.bundle / "submission"
    out = (args.out or args.bundle.parent / manifest["zip_name"]).expanduser()

    for name, spec in manifest["encoders"].items():
        print(f"[{name}] {spec['repo']} @ {spec['revision'][:10]}", flush=True)
        hub = load_file(hf_hub_download(spec["repo"], "model.safetensors",
                                        revision=spec["revision"]))
        tensors, bad = {}, []
        for key, info in spec["tensors"].items():
            if key not in hub:
                sys.exit(f"  missing tensor {key} in {spec['repo']}")
            t = np.ascontiguousarray(hub[key].astype(info["dtype"]))
            if list(t.shape) != info["shape"] or \
                    hashlib.sha256(t.tobytes()).hexdigest() != info["sha256"]:
                bad.append(key)
            tensors[key] = t
        if bad:
            sys.exit(f"  {len(bad)} tensors differ from the original ZIP, e.g. {bad[:3]}; "
                     "not writing a ZIP")
        target = sub / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(target), metadata=spec["metadata"])
        print(f"  {len(tensors)} tensors verified -> {spec['path']}", flush=True)

    missing = [p for p in manifest["files"] if not (sub / p).is_file()]
    if missing:
        sys.exit(f"bundle incomplete, missing {missing[:5]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in manifest["files"]:
            zf.write(sub / p, p)
    print(f"Wrote {out} ({out.stat().st_size / 2**20:.1f} MiB, {len(manifest['files'])} files). "
          "All encoder tensors match the original ZIP.")


if __name__ == "__main__":
    main()
