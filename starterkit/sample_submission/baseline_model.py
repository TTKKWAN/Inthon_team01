"""Frozen DINOv2 / MPNet encoders and a concat -> MLP binary matching model.

This file is also exported with the submission. Change the model here so that
training and inference always use the same implementation.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

IMAGE_MODEL_ID = "facebook/dinov2-base"
TEXT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"


class SimilarityMLP(nn.Module):
    """One scalar in [0, 1] for each concatenated image/text pair."""

    def __init__(self, image_dim=768, text_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, image_features, text_features):
        pairs = torch.cat([image_features, text_features], dim=-1)
        return self.network(pairs).squeeze(-1)

    def forward(self, image_features, text_features):
        return self.logits(image_features, text_features).sigmoid()


class FrozenEncoders:
    def __init__(
        self, image_model_dir, text_model_dir, device="cpu",
        image_batch_size=16, text_batch_size=32, max_text_length=384,
    ):
        self.device = torch.device(device)
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.max_text_length = max_text_length
        if min(image_batch_size, text_batch_size, max_text_length) < 1:
            raise ValueError("Batch sizes and max_text_length must be positive.")
        if max_text_length > 384:
            raise ValueError("The baseline MPNet encoder supports at most 384 text tokens.")
        # All paths are local. Evaluation never downloads models or tokenizers.
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_model_dir, local_files_only=True, use_fast=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_dir, local_files_only=True,
        )
        self.image_model = AutoModel.from_pretrained(
            image_model_dir, local_files_only=True, dtype=torch.float32,
        )
        self.text_model = AutoModel.from_pretrained(
            text_model_dir, local_files_only=True, dtype=torch.float32,
        )
        if self.image_model.config.model_type != "dinov2":
            raise ValueError(f"Use the allowed image model: {IMAGE_MODEL_ID}")
        if self.text_model.config.model_type != "mpnet":
            raise ValueError(f"Use the allowed text model: {TEXT_MODEL_ID}")
        self.image_dim = self.image_model.config.hidden_size
        self.text_dim = self.text_model.config.hidden_size
        # CPU inference uses float32. CUDA inference uses float16 to reduce memory.
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        for model in (self.image_model, self.text_model):
            model.requires_grad_(False).to(device=self.device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def encode_images(self, images):
        """Encode every original frame, then mean-pool across frames."""
        if not images:
            raise ValueError("An episode must contain at least one image.")
        total = torch.zeros(self.image_dim, device=self.device, dtype=torch.float32)
        for start in range(0, len(images), self.image_batch_size):
            inputs = self.image_processor(
                images=[image.convert("RGB") for image in images[start:start + self.image_batch_size]],
                return_tensors="pt",
            )
            pixels = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            # DINOv2's CLS token is the embedding of a single frame.
            features = self.image_model(pixel_values=pixels).last_hidden_state[:, 0]
            total += features.float().sum(dim=0)
        return (total / len(images)).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts):
        """Attention-mask-aware mean pooling of MPNet token embeddings."""
        if not texts:
            return torch.empty(0, self.text_dim)
        features = []
        for start in range(0, len(texts), self.text_batch_size):
            inputs = self.tokenizer(
                texts[start:start + self.text_batch_size], padding=True,
                truncation=True, max_length=self.max_text_length, return_tensors="pt",
            ).to(self.device)
            tokens = self.text_model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            features.append(pooled.cpu())
        return torch.cat(features)

    def export(self, destination):
        """Export only the frozen encoders and processors needed for inference."""
        destination = Path(destination)
        for name, model in [("dinov2", self.image_model), ("mpnet", self.text_model)]:
            folder = destination / name
            # Half-precision storage halves the artifact size. CPU load casts to float32.
            state = {key: value.detach().cpu().half() if value.is_floating_point()
                     else value.detach().cpu() for key, value in model.state_dict().items()}
            model.save_pretrained(folder, state_dict=state, safe_serialization=True)
        self.image_processor.save_pretrained(destination / "dinov2")
        self.tokenizer.save_pretrained(destination / "mpnet")
