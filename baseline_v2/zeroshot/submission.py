"""SigLIP zero-shot control (no training): server API v2.

score(choice) = mean over frames of cos(SigLIP image(frame), SigLIP text(choice)), where the
choice's text embedding is the normalized mean of its clause embeddings (the clause rules of
baseline_v2 model_v2.split_clauses), i.e. eval_nnval.py's `siglip_zs`. Every call uses only
that episode's images and choices; nothing is stored between calls.
"""

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, PreTrainedTokenizerFast

MAX_CLAUSES = 8
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(
    r",\s*(?:and\s+)?then\s+|;\s*|\s+and\s+then\s+|,\s*(?:before|after which|followed by)\s+",
    re.IGNORECASE,
)


def split_clauses(text, max_clauses=MAX_CLAUSES):
    parts = []
    for sentence in _SENTENCE.split(text.strip()):
        parts.extend(p.strip(" ,.") for p in _CLAUSE.split(sentence))
    parts = [p for p in parts if p] or [text.strip()]
    if len(parts) > max_clauses:
        parts = parts[:max_clauses - 1] + [" ".join(parts[max_clauses - 1:])]
    return parts


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir) / "siglip"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True,
                                                            use_fast=False)
        # tokenizer.json through `tokenizers` only: the grading image has no sentencepiece.
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tokenizer.json"), pad_token="</s>", eos_token="</s>",
            unk_token="<unk>", model_input_names=["input_ids"])
        self.model = AutoModel.from_pretrained(root, local_files_only=True, dtype=self.dtype)
        if self.model.config.model_type != "siglip":
            raise ValueError("Use the allowed model: google/siglip-base-patch16-224")
        self.model.requires_grad_(False).to(self.device).eval()

    @torch.inference_mode()
    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        pixels = self.processor(images=[im.convert("RGB") for im in images],
                                return_tensors="pt")["pixel_values"]
        frames = F.normalize(self.model.get_image_features(
            pixel_values=pixels.to(self.device, self.dtype)).float(), dim=-1)
        clause_lists = [split_clauses(choice["text"]) for choice in choices]
        flat = [c.lower() for clauses in clause_lists for c in clauses]
        # SigLIP was trained on fixed 64-token padding without an attention mask.
        ids = self.tokenizer(flat, padding="max_length", truncation=True, max_length=64,
                             return_tensors="pt")["input_ids"].to(self.device)
        emb = F.normalize(self.model.get_text_features(input_ids=ids).float(), dim=-1)
        texts, start = [], 0
        for clauses in clause_lists:
            texts.append(F.normalize(emb[start:start + len(clauses)].mean(0), dim=-1))
            start += len(clauses)
        scores = (torch.stack(texts) @ frames.T).mean(1)
        if not torch.isfinite(scores).all():
            raise ValueError("Invalid scores.")
        return scores.cpu().tolist()

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]["id"] for i in order]
