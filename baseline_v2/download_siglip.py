"""Download the allowed image-text encoder google/siglip-base-patch16-224 to models/siglip.

DINOv2 / MPNet come from the starter kit: `cd ../starterkit && python download_models.py`.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    target = ROOT / "models" / "siglip"
    snapshot_download("google/siglip-base-patch16-224", local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    print(f"Ready: google/siglip-base-patch16-224 -> {target}")
