"""Cache DINOv2 patch tokens pooled to a 4x4 grid for every Train frame.

Same episodes, frame order and frame_offsets as baseline_v2/cache/frames.pt (prepare_features.py),
so row i of both caches is the same frame. Output: baseline_v2/cache/patches4.pt with
episode_ids, frame_offsets, patches [n_frames, 16, 768] fp16 (row-major 4x4 grid).

    python diff_probe/cache_patches.py --check episode_144 episode_169
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline_v2"))
from prepare_features import load_episodes  # noqa: E402

GRID = 4


class Frames(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.processor(images=[im.convert("RGB")], return_tensors="pt")["pixel_values"][0]


@torch.inference_mode()
def encode(model, px):
    hidden = model(pixel_values=px).last_hidden_state  # [B, 1 + 16*16, 768]
    side = int((hidden.shape[1] - 1) ** 0.5)
    grid = hidden[:, 1:].transpose(1, 2).reshape(len(px), -1, side, side)
    pooled = F.adaptive_avg_pool2d(grid.float(), GRID).flatten(2).transpose(1, 2)  # [B, 16, 768]
    return hidden[:, 0].float(), pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "baseline_v2/config-a100.json")
    ap.add_argument("--out", type=Path, default=ROOT / "baseline_v2/cache/patches4.pt")
    ap.add_argument("--check", nargs="*", default=[], help="episodes whose CLS is compared to frames.pt")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    base = args.config.resolve().parent
    model_dir = base / cfg["image_model_dir"]
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    if model.config.model_type != "dinov2":
        raise ValueError("expected facebook/dinov2-base")
    model = model.to("cuda:0", torch.float16).eval()

    episodes = load_episodes(base / cfg["data_dir"])
    ref = torch.load(ROOT / "baseline_v2/cache/frames.pt", map_location="cpu", weights_only=False)
    assert [e for e, _ in episodes] == ref["episode_ids"], "episode order differs from frames.pt"
    offsets = [0]
    for _, frames in episodes:
        offsets.append(offsets[-1] + len(frames))
    assert torch.equal(torch.tensor(offsets), ref["frame_offsets"]), "frame counts differ from frames.pt"

    paths = [p for _, frames in episodes for p in frames]
    loader = DataLoader(Frames(paths, processor), batch_size=256, num_workers=cfg.get("num_workers", 8))
    cls_all, out = [], []
    for n, px in enumerate(loader):
        cls, pooled = encode(model, px.to("cuda:0", torch.float16))
        cls_all.append(cls.cpu())
        out.append(pooled.half().cpu())
        if n % 40 == 0:
            print(f"frames {min((n + 1) * 256, len(paths))}/{len(paths)}", flush=True)
    cls_all = torch.cat(cls_all)
    cos = F.cosine_similarity(cls_all, ref["frames"][:, :768].float(), dim=-1)
    print(f"CLS vs frames.pt cos: min {cos.min():.4f} mean {cos.mean():.4f}")
    index = {e: i for i, e in enumerate(ref["episode_ids"])}
    for e in args.check:
        s, t = offsets[index[e]], offsets[index[e] + 1]
        a = F.normalize(cls_all[s:t], dim=-1)
        b = F.normalize(ref["frames"][s:t, :768].float(), dim=-1)
        sim = a @ b.T
        print(e, "frames", t - s, "diag min", round(sim.diag().min().item(), 4),
              "argmax==diag", bool((sim.argmax(1) == torch.arange(t - s)).all()))
    torch.save({"episode_ids": ref["episode_ids"], "frame_offsets": ref["frame_offsets"],
                "patches": torch.cat(out)}, args.out)
    print("saved", args.out, tuple(torch.cat(out).shape))


if __name__ == "__main__":
    main()
