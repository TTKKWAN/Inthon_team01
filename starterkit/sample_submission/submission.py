"""Server API v2: load once, score every candidate, return all IDs in descending order."""

import json
from pathlib import Path

import torch
from baseline_model import FrozenEncoders, SimilarityMLP


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(config.get("cpu_threads", 4))
        self.device = torch.device(device)
        self.encoders = FrozenEncoders(
            root / "models/dinov2", root / "models/mpnet", device,
            config["image_batch_size"], config["text_batch_size"], config["max_text_length"],
        )
        self.model = SimilarityMLP(**config["model_config"]).to(self.device)
        state = torch.load(root / "weights/mlp.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def score(self, images, choices):
        """One sigmoid matching score in [0, 1] per candidate; no episode state is stored."""
        if not choices:
            raise ValueError("choices must not be empty.")
        video = self.encoders.encode_images(images).to(self.device)
        texts = self.encoders.encode_texts([choice["text"] for choice in choices]).to(self.device)
        scores = self.model(video.expand(len(choices), -1), texts)
        if scores.ndim != 1 or not torch.isfinite(scores).all():
            raise ValueError("Invalid similarity scores.")
        return scores.cpu().tolist()

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda index: -scores[index])
        return [choices[index]["id"] for index in order]
