"""Download the Train data from the team Hugging Face dataset repo and extract it.

Usage (from the repo root):
    pip install huggingface_hub
    python scripts/download_data.py            # -> data/Train/
    python scripts/download_data.py --out-dir /content/data

Offline helper only. Do not import this from inference/submission code.
"""

import argparse
import hashlib
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "TTaekwan/Inthon_team01_public"
ARCHIVE = "Train.tar"
CHECKSUM = "Train.tar.sha256"
ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--keep-archive", action="store_true", help="keep the HF cache copy of Train.tar")
    args = parser.parse_args()

    target = args.out_dir / "Train"
    if target.exists() and any(target.iterdir()):
        print(f"{target} already exists; nothing to do.")
        return

    archive = Path(hf_hub_download(REPO_ID, ARCHIVE, repo_type="dataset", revision=args.revision))
    expected = Path(hf_hub_download(REPO_ID, CHECKSUM, repo_type="dataset", revision=args.revision))
    expected = expected.read_text().split()[0]
    actual = sha256(archive)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {ARCHIVE}: expected {expected}, got {actual}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(args.out_dir, filter="data")
        else:  # Python < 3.11.4
            tar.extractall(args.out_dir)
    episodes = sum(1 for p in target.iterdir() if p.name.startswith("episode_"))
    print(f"Extracted {episodes} episodes to {target}")

    if not args.keep_archive:
        archive.resolve().unlink(missing_ok=True)


if __name__ == "__main__":
    main()
