"""Original Train episodes, episode-level splits, and one random negative per episode."""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Episode:
    episode_id: str
    frames: tuple[Path, ...]
    description: str


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", Path(path).name)]


def load_episodes(data_dir):
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Train directory does not exist: {root}")
    episodes = []
    for folder in sorted(root.glob("episode_*"), key=natural_key):
        if not folder.is_dir():
            continue
        description = (folder / "description.txt").read_text(encoding="utf-8").strip()
        frames = tuple(sorted(
            (p for p in (folder / "images").iterdir()
             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}),
            key=natural_key,
        ))
        if not description or not frames:
            raise ValueError(f"Missing description or images: {folder}")
        episodes.append(Episode(folder.name, frames, description))
    if not episodes:
        raise ValueError(f"No episode_*/description.txt episodes found in {root}")
    return episodes


def load_images(episode):
    images = []
    for path in episode.frames:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def split_indices(episodes, validation_fraction=0.2, seed=2026):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(episodes) < 4:
        raise ValueError("At least four episodes are needed: two per split.")
    indices = list(range(len(episodes)))
    random.Random(seed).shuffle(indices)
    size = min(len(episodes) - 2, max(2, round(len(episodes) * validation_fraction)))
    return sorted(indices[size:]), sorted(indices[:size])


def feature_signature(episodes, config):
    """Invalidate cached features when source data or encoder files change."""
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode())
        digest.update(episode.description.encode())
        for path in episode.frames:
            stat = path.stat()
            digest.update(str((str(path.resolve()), stat.st_size, stat.st_mtime_ns)).encode())
    for key in ["image_model_dir", "text_model_dir"]:
        root = Path(config[key])
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                stat = path.stat()
                digest.update(str((str(path.resolve()), stat.st_size, stat.st_mtime_ns)).encode())
    digest.update(str(config["max_text_length"]).encode())
    digest.update(b"dinov2-cls-frame-mean_mpnet-masked-token-mean_v1")
    return digest.hexdigest()


def extract_features(episodes, encoders, cache_path, signature):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("signature") != signature:
            raise ValueError(f"Feature cache is stale; use a new output directory: {cache_path}")
        print(f"Loaded cached features: {cache_path}", flush=True)
        return cached["image_features"], cached["text_features"]
    image_features = []
    for index, episode in enumerate(episodes):
        images = load_images(episode)
        try:
            image_features.append(encoders.encode_images(images))
        finally:
            for image in images:
                image.close()
        if (index + 1) % 50 == 0 or index + 1 == len(episodes):
            print(f"Encoded image sequences: {index + 1}/{len(episodes)}", flush=True)
    image_features = torch.stack(image_features)
    text_features = encoders.encode_texts([e.description for e in episodes])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "image_features": image_features,
                "text_features": text_features}, cache_path)
    return image_features, text_features


class EpisodePairs(Dataset):
    """Each item is exactly (own description, label=1), (random description, label=0).

    Negative descriptions are sampled only from this dataset's episode split.
    They are resampled for every training epoch; validation stays at epoch zero.
    """

    def __init__(self, image_features, text_features, descriptions, seed=2026):
        if len(image_features) != len(text_features) or len(descriptions) != len(text_features):
            raise ValueError("Image, text and description counts must match.")
        if len(set(descriptions)) < 2:
            raise ValueError("Each split needs at least two distinct descriptions.")
        self.image_features = image_features
        self.text_features = text_features
        self.descriptions = descriptions
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        self.negative_indices = []
        for index, text in enumerate(self.descriptions):
            negative = rng.randrange(len(self.descriptions))
            while negative == index or self.descriptions[negative] == text:
                negative = rng.randrange(len(self.descriptions))
            self.negative_indices.append(negative)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, index):
        negative = self.negative_indices[index]
        return (
            self.image_features[index].expand(2, -1),
            torch.stack([self.text_features[index], self.text_features[negative]]),
            torch.tensor([1.0, 0.0]),
        )


def write_splits(path, episodes, train, valid):
    Path(path).write_text(json.dumps({
        "train": [episodes[i].episode_id for i in train],
        "valid": [episodes[i].episode_id for i in valid],
    }, indent=2) + "\n", encoding="utf-8")
