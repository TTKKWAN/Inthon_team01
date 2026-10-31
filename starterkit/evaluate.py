"""Evaluate original positive / fixed random-negative pairs on the saved validation split."""

import argparse
import json
from pathlib import Path

import torch
from baseline_model import SimilarityMLP
from common import choose_device, seed_everything
from dataset import feature_signature, load_episodes
from torch.utils.data import DataLoader
from train import make_pairs, run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    seed_everything(config["seed"], config["cpu_threads"])
    device = choose_device(args.device)
    splits = json.loads((args.checkpoint.parent / "splits.json").read_text())
    all_episodes = {episode.episode_id: episode for episode in load_episodes(config["data_dir"])}
    selected_ids = set(splits["train"] + splits["valid"])
    episodes = [e for e in all_episodes.values() if e.episode_id in selected_ids]
    features = torch.load(args.checkpoint.parent / "features.pt", map_location="cpu", weights_only=True)
    if feature_signature(episodes, config) != features["signature"]:
        raise ValueError("Validation source data or encoders changed since feature extraction.")
    valid_ids = set(splits["valid"])
    indices = [i for i, e in enumerate(episodes) if e.episode_id in valid_ids]
    pairs = make_pairs(features["image_features"], features["text_features"], episodes,
                       indices, config["seed"] + 1)
    model = SimilarityMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = run_epoch(model, DataLoader(pairs, batch_size=config["batch_size"]), device)
    print(json.dumps({"validation": metrics,
                      "note": "Random-negative pair metrics; not leaderboard accuracy."}, indent=2))


if __name__ == "__main__":
    main()
