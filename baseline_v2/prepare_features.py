"""Encode Train once with the frozen encoders and cache the results.

  --part frames : cache/frames.pt
      episode_ids [N], frames [sum(T), 2304] fp16, frame_offsets [N+1]
  --part texts  : cache/texts.pt  (needs cache/negatives.json from negatives.py)
      texts [M] unique strings, full [M, 1536] fp16,
      clause_flat [sum(K), 1536] fp16, clause_offsets [M+1],
      groups: per episode, list of {"pos": text id, "negs": [[type, text id], ...]}

Images and texts are cached separately so a new negative bank only re-encodes text.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model_v2 import FrozenEncoders, pick_device

ROOT = Path(__file__).resolve().parent


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", Path(path).name)]


def load_episodes(data_dir):
    episodes = []
    for folder in sorted(Path(data_dir).glob("episode_*"), key=natural_key):
        if not folder.is_dir():
            continue
        frames = sorted((p for p in (folder / "images").iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
                        key=natural_key)
        episodes.append((folder.name, frames))
    return episodes


class FrameDataset(Dataset):
    def __init__(self, paths, encoders):
        self.paths, self.encoders = paths, encoders

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            dino, sig = self.encoders.preprocess([im])
        return dino[0], sig[0]


def encode_frames(cfg, base, enc, max_episodes, out):
    episodes = load_episodes(base / cfg["data_dir"])[:max_episodes]
    paths = [p for _, frames in episodes for p in frames]
    offsets = [0]
    for _, frames in episodes:
        offsets.append(offsets[-1] + len(frames))
    # Workers only need the CPU-side processors; keep the GPU models out of the pickled copy.
    processors = FrozenEncoders.__new__(FrozenEncoders)
    processors.dino_processor, processors.siglip_processor = (enc.dino_processor,
                                                              enc.siglip_processor)
    loader = DataLoader(FrameDataset(paths, processors), batch_size=128,
                        num_workers=cfg.get("num_workers", 6), pin_memory=True)
    chunks = []
    for n, (dino_px, sig_px) in enumerate(loader):
        chunks.append(enc.encode_pixels(dino_px, sig_px).half().cpu())
        if n % 40 == 0:
            print(f"frames {min((n + 1) * 128, len(paths))}/{len(paths)}", flush=True)
    torch.save({"episode_ids": [e for e, _ in episodes], "frames": torch.cat(chunks),
                "frame_offsets": torch.tensor(offsets)}, out)
    print(f"Saved {len(episodes)} episodes, {len(paths)} frames -> {out}", flush=True)


def encode_texts(cfg, base, enc, episode_ids, out):
    bank = json.loads((base / cfg["negative_bank"]).read_text(encoding="utf-8"))["episodes"]
    texts, index = [], {}

    def tid(text):
        if text not in index:
            index[text] = len(texts)
            texts.append(text)
        return index[text]

    groups = [[{"pos": tid(g["pos"]), "negs": [[k, tid(t)] for k, t in g["negs"]]}
               for g in bank[e]] for e in episode_ids]
    full, clause_flat, offsets = [], [], [0]
    for s in range(0, len(texts), 2048):
        chunk = texts[s:s + 2048]
        f, clauses, mask = enc.encode_descriptions(chunk)
        full.append(f.half())
        for c, m in zip(clauses, mask):
            clause_flat.append(c[m].half())
            offsets.append(offsets[-1] + int(m.sum()))
        print(f"texts {s + len(chunk)}/{len(texts)}", flush=True)
    torch.save({"texts": texts, "full": torch.cat(full), "clause_flat": torch.cat(clause_flat),
                "clause_offsets": torch.tensor(offsets), "groups": groups,
                "episode_ids": episode_ids}, out)
    mean_k = (offsets[-1] / len(texts))
    print(f"Saved {len(texts)} texts ({mean_k:.2f} clauses each) -> {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--part", choices=["frames", "texts", "all"], default="all")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--suffix", default="", help="e.g. _smoke -> cache/frames_smoke.pt")
    parser.add_argument("--texts-suffix", help="texts cache suffix if different from --suffix")
    parser.add_argument("--bank", help="negative bank path (relative to config), overrides config")
    parser.add_argument("--clause-lexicon",
                        help="clause_split lexicon JSON (relative to config); default: split_clauses")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    if args.bank:
        cfg["negative_bank"] = args.bank
    cache = base / cfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    frames_out = cache / f"frames{args.suffix}.pt"
    texts_suffix = args.suffix if args.texts_suffix is None else args.texts_suffix
    texts_out = cache / f"texts{texts_suffix}.pt"
    for path, part in [(frames_out, "frames"), (texts_out, "texts")]:
        if args.part in (part, "all") and path.exists():
            raise FileExistsError(f"Cache exists, delete it first: {path}")
    device = pick_device()
    lexicon = base / args.clause_lexicon if args.clause_lexicon else None
    enc = FrozenEncoders(base / cfg["image_model_dir"], base / cfg["text_model_dir"],
                         base / cfg["siglip_model_dir"], device, clause_lexicon=lexicon,
                         max_clauses=cfg.get("max_clauses", 8))
    if args.part in ("frames", "all"):
        encode_frames(cfg, base, enc, args.max_episodes, frames_out)
    if args.part in ("texts", "all"):
        ids = torch.load(frames_out, weights_only=False)["episode_ids"]
        encode_texts(cfg, base, enc, ids, texts_out)


if __name__ == "__main__":
    main()
