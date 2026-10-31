"""Server API v2: load once, score every candidate, return all IDs in descending order."""

import json
from pathlib import Path

import torch
from model_v2 import FrozenEncoders, OrderAwareScorer


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        config = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(config.get("cpu_threads", 4))
        self.device = torch.device(device)
        self.encoders = FrozenEncoders(root / "models/dinov2", root / "models/mpnet",
                                       root / "models/siglip", device)
        self.model = OrderAwareScorer(**config["model_config"]).to(self.device)
        state = torch.load(root / "weights/scorer.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        frames = self.encoders.encode_frames(images).to(self.device)
        full, clauses, cmask = self.encoders.encode_descriptions([c["text"] for c in choices])
        n = len(choices)
        logits = self.model(
            frames.unsqueeze(0).expand(n, -1, -1),
            torch.ones(n, frames.shape[0], dtype=torch.bool, device=self.device),
            full.to(self.device), clauses.to(self.device), cmask.to(self.device),
        )
        if not torch.isfinite(logits).all():
            raise ValueError("Invalid scores.")
        return logits.sigmoid().cpu().tolist()

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]["id"] for i in order]
