"""Server API v2: marked-token verifier (shipped as submission.py).

Per episode: encode every frame with the frozen DINOv2 + SigLIP image encoders (the same
features the head was trained on), encode the five choices with frozen MPNet, mark the
words where the choices differ (rule-based, marks.py), score each choice from its marked
tokens against the frame sequence, and return all IDs best first.
Only this episode's images and choices are used.
"""

import json
from pathlib import Path

import torch
from marked_model import MarkedVerifier
from marks import mark_spans
from model_v2 import FrozenEncoders


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        cfg = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(cfg.get("cpu_threads", 4))
        self.device = torch.device(device)
        self.enc = FrozenEncoders(root / "models/dinov2", root / "models/mpnet",
                                  root / "models/siglip", device)
        self.model = MarkedVerifier(**cfg["model_config"]).to(self.device).eval()
        state = torch.load(root / "weights/verifier.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.max_frames = cfg["max_frames"]
        self.marks = cfg["marks"]
        self.max_tokens = cfg["max_text_tokens"]

    @torch.inference_mode()
    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        texts = [c["text"] for c in choices]

        feats = self.enc.encode_frames(images)
        if len(feats) > self.max_frames:
            idx = [round(k * (len(feats) - 1) / (self.max_frames - 1)) for k in range(self.max_frames)]
            feats = feats[idx]
        feats = feats.half().float()  # the training cache stored fp16 features
        fx = feats.unsqueeze(0).to(self.device)
        fm = torch.ones(1, len(feats), dtype=torch.bool, device=self.device)
        mem = self.model.encode_video(fx, fm)

        tok = self.enc.tokenizer(texts, return_offsets_mapping=True, return_tensors="pt",
                                 padding=True, truncation=True, max_length=self.max_tokens)
        offsets = tok.pop("offset_mapping")
        marks = torch.zeros(offsets.shape[:2], dtype=torch.long)
        for n, spans in enumerate(mark_spans(texts, self.marks)):
            for s, e in spans:
                hit = ((offsets[n, :, 1] > s) & (offsets[n, :, 0] < e)
                       & (offsets[n, :, 1] > offsets[n, :, 0]))
                marks[n][hit] = 1
        tok = tok.to(self.device)
        hidden = self.enc.text_model(**tok).last_hidden_state.float()
        n = len(texts)
        scores = self.model.score(hidden, tok["attention_mask"].bool(), marks.to(self.device),
                                  mem.expand(n, -1, -1), fm.expand(n, -1))
        if not torch.isfinite(scores).all():
            raise ValueError("Invalid scores.")
        return scores.float().cpu().tolist()

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]["id"] for i in order]
