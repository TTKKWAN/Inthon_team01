"""One original positive + one random negative per episode; train only the concat MLP."""

import argparse
import json
import random
from pathlib import Path

import torch
from baseline_model import IMAGE_MODEL_ID, TEXT_MODEL_ID, FrozenEncoders, SimilarityMLP
from common import ROOT, choose_device, load_config, save_json, seed_everything
from dataset import (
    EpisodePairs,
    extract_features,
    feature_signature,
    load_episodes,
    split_indices,
    write_splits,
)
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count, binary_correct, pair_correct = 0.0, 0, 0, 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.set_grad_enabled(training):
        for images, texts, labels in loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            logits = model.logits(images, texts)  # [episodes, 2], own text first.
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.numel()
            count += labels.numel()
            binary_correct += ((logits >= 0) == labels.bool()).sum().item()
            pair_correct += (logits[:, 0] > logits[:, 1]).sum().item()
    return {
        "bce": loss_sum / count,
        "binary_accuracy": binary_correct / count,
        "pair_accuracy": pair_correct / (count // 2),
        "episodes": count // 2,
    }


def make_pairs(image_features, text_features, episodes, indices, seed):
    return EpisodePairs(image_features[indices], text_features[indices],
                        [episodes[i].description for i in indices], seed=seed)


def train(config, device="auto", max_episodes=None):
    seed_everything(config["seed"], config["cpu_threads"])
    device = choose_device(device)
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["patience"] < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    episodes = load_episodes(config["data_dir"])
    total_episodes = len(episodes)
    if max_episodes is not None:
        if max_episodes < 4:
            raise ValueError("--max-episodes must be at least four.")
        picked = random.Random(config["seed"]).sample(
            range(len(episodes)), min(max_episodes, len(episodes)),
        )
        episodes = [episodes[i] for i in sorted(picked)]
    train_indices, valid_indices = split_indices(
        episodes, config["validation_fraction"], config["seed"],
    )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Do not silently replace an existing run/checkpoint or mix experiments.
    if (output / "best.pt").exists() or (output / "history.json").exists():
        raise FileExistsError(f"Use a new --output-dir; an existing run is present: {output}")
    save_json(output / "config.json", config)
    write_splits(output / "splits.json", episodes, train_indices, valid_indices)
    print(f"Device: {device}; train: {len(train_indices)}; valid: {len(valid_indices)}", flush=True)
    encoders = FrozenEncoders(
        config["image_model_dir"], config["text_model_dir"], device,
        config["image_batch_size"], config["text_batch_size"], config["max_text_length"],
    )
    signature = feature_signature(episodes, config)
    images, texts = extract_features(episodes, encoders, output / "features.pt", signature)
    model_config = {
        "image_dim": encoders.image_dim, "text_dim": encoders.text_dim,
        "hidden_dim": config["hidden_dim"], "dropout": config["dropout"],
    }
    del encoders
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # Clone to normal tensors: cached encoder outputs never participate in encoder gradients.
    images, texts = images.clone(), texts.clone()
    train_pairs = make_pairs(images, texts, episodes, train_indices, config["seed"])
    valid_pairs = make_pairs(images, texts, episodes, valid_indices, config["seed"] + 1)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_pairs, batch_size=config["batch_size"],
                              shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_pairs, batch_size=config["batch_size"], shuffle=False)
    model = SimilarityMLP(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    best_loss, stale, history = float("inf"), 0, []
    provenance = {
        "image_model_id": IMAGE_MODEL_ID, "text_model_id": TEXT_MODEL_ID,
        "total_available_episodes": total_episodes, "used_episodes": len(episodes),
        "train_episodes": len(train_indices), "valid_episodes": len(valid_indices),
        "demo_subset": len(episodes) < total_episodes, "feature_signature": signature,
        "negative_sampling": "one random original description from the same split per episode",
    }
    for epoch in range(config["epochs"]):
        train_pairs.set_epoch(epoch)
        metrics = {"epoch": epoch + 1,
                   "train": run_epoch(model, train_loader, device, optimizer),
                   "valid": run_epoch(model, valid_loader, device)}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["valid"]["bce"] < best_loss:
            best_loss, stale = metrics["valid"]["bce"], 0
            torch.save({
                "format_version": 1, "model_config": model_config,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config, "epoch": epoch + 1, "metrics": metrics,
                "provenance": provenance,
            }, output / "best.pt")
        else:
            stale += 1
        save_json(output / "history.json", history)
        if stale >= config["patience"]:
            print("Early stopping.", flush=True)
            break
    save_json(output / "provenance.json", provenance)
    print(f"Saved best checkpoint: {output / 'best.pt'}", flush=True)
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, help="Optional small subset for a smoke run.")
    parser.add_argument("--epochs", type=int)
    for name in ["data-dir", "output-dir", "image-model-dir", "text-model-dir"]:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["data_dir", "output_dir", "image_model_dir", "text_model_dir"]:
        if getattr(args, key) is not None:
            config[key] = str(getattr(args, key).resolve())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    train(config, args.device, args.max_episodes)


if __name__ == "__main__":
    main()
