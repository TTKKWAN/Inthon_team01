"""Server API v2: span centrality + two video judges with their text prior removed.

score(choice) = span centrality (byte distance breaks ties)
              + l1 * z(align - align prior) + l2 * z(head - head prior)
  align : SigLIP clauses matched to frames in order (monotonic), mean similarity (zero-shot)
  head  : small head over frozen [DINOv2 CLS ; DINOv2 mean patch ; SigLIP] frames
          (head_type "pair"), plus each frame's DINOv2 patches pooled to 4x4 ("patch")
  prior : the same score averaged over 64 fixed Train reference videos (weights/refs.pt),
          i.e. how much the wording alone scores; subtracting it leaves the video's share
  z     : standardized over this episode's choices
Only this episode's images and choices are used; nothing is learned or stored at test time.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from centrality import _mean_scores, score_span
from hyuwooma_bytes import normalized_byte_distance
from judge import GRID_TOKENS, MAX_CLAUSES, PairHead, PatchHead, batched_align
from model_v2 import SIGLIP_DIM, FrozenEncoders, _load, load_siglip_tokenizer, split_clauses
from transformers import AutoImageProcessor


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6)


class Submission:
    def __init__(self, artifact_dir: str, device: str):
        root = Path(artifact_dir)
        cfg = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(cfg.get("cpu_threads", 4))
        self.device = torch.device(device)
        self.l1, self.l2, self.tmax = cfg["l1"], cfg["l2"], cfg["tmax"]
        self.patch = cfg.get("head_type", "pair") == "patch"

        # Image encoders only (no MPNet): same preprocessing and features as the training cache.
        enc = FrozenEncoders.__new__(FrozenEncoders)
        enc.device, enc.image_batch_size, enc.text_batch_size = self.device, 32, 64
        enc.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        dino_dir, siglip_dir = root / cfg["dinov2_dir"], root / cfg["siglip_dir"]
        enc.dino_processor = AutoImageProcessor.from_pretrained(dino_dir, local_files_only=True, use_fast=False)
        enc.siglip_processor = AutoImageProcessor.from_pretrained(siglip_dir, local_files_only=True, use_fast=False)
        enc.siglip_tokenizer = load_siglip_tokenizer(siglip_dir)
        enc.image_model, enc.siglip = _load(dino_dir), _load(siglip_dir)
        if enc.image_model.config.model_type != "dinov2" or enc.siglip.config.model_type != "siglip":
            raise ValueError("Use the allowed models: facebook/dinov2-base, google/siglip-base-patch16-224")
        for model in (enc.image_model, enc.siglip):
            model.requires_grad_(False).to(device=self.device, dtype=enc.dtype).eval()
        self.enc = enc

        self.head = (PatchHead if self.patch else PairHead)(tmax=self.tmax).to(self.device).eval()
        self.head.load_state_dict(torch.load(root / "weights/head.pt", map_location="cpu", weights_only=True))
        saved = torch.load(root / "weights/refs.pt", map_location="cpu", weights_only=True)
        refs = saved["frames"]
        x = torch.zeros(len(refs), self.tmax, refs[0].shape[1])
        mask = torch.ones(len(refs), self.tmax, dtype=torch.bool)
        for i, f in enumerate(refs):
            x[i, :len(f)], mask[i, :len(f)] = f.float(), False
        self.ref_mask = mask.to(self.device)
        patches = None
        if self.patch:
            patches = torch.zeros(len(refs), self.tmax, GRID_TOKENS, saved["patches"][0].shape[-1])
            for i, p in enumerate(saved["patches"]):
                patches[i, :len(p)] = p.float()
            patches = patches.to(self.device)
        with torch.inference_mode():
            self.ref_video, self.ref_vmask = self._video(x.to(self.device), self.ref_mask, patches)
        self.ref_sig = F.normalize(x[..., -SIGLIP_DIM:], dim=-1).to(self.device)

    def _video(self, frames, fmask, patches):
        if self.patch:
            return self.head.encode_video(frames, fmask, patches)
        return self.head.encode_video(frames, fmask), fmask

    @torch.inference_mode()
    def _encode(self, images):
        """[T, 2304] frame features exactly as FrozenEncoders.encode_pixels, and the same DINOv2
        pass's patch tokens pooled to a 4x4 grid [T, 16, 768] (diff_probe/cache_patches.py)."""
        enc, feats, grids = self.enc, [], []
        for s in range(0, len(images), enc.image_batch_size):
            dino_px, sig_px = enc.preprocess(images[s:s + enc.image_batch_size])
            hidden = enc.image_model(pixel_values=dino_px.to(self.device, enc.dtype)).last_hidden_state
            sig = enc.siglip.get_image_features(pixel_values=sig_px.to(self.device, enc.dtype))
            feats.append(torch.cat([hidden[:, 0], hidden[:, 1:].mean(1), F.normalize(sig, dim=-1)], -1).float())
            side = int((hidden.shape[1] - 1) ** 0.5)
            grid = hidden[:, 1:].transpose(1, 2).reshape(len(hidden), -1, side, side)
            grids.append(F.adaptive_avg_pool2d(grid.float(), 4).flatten(2).transpose(1, 2))
        return torch.cat(feats), torch.cat(grids)

    def _clauses(self, texts):
        lists = [split_clauses(t)[:MAX_CLAUSES] for t in texts]
        flat = list(dict.fromkeys(c for cl in lists for c in cl))
        emb = dict(zip(flat, self.enc.encode_siglip_text(flat)))
        x = torch.zeros(len(texts), MAX_CLAUSES, SIGLIP_DIM)
        mask = torch.ones(len(texts), MAX_CLAUSES, dtype=torch.bool)
        for i, cl in enumerate(lists):
            x[i, :len(cl)] = torch.stack([emb[c] for c in cl])
            mask[i, :len(cl)] = False
        return x.to(self.device), mask.to(self.device)

    @torch.inference_mode()
    def score(self, images, choices):
        if not choices:
            raise ValueError("choices must not be empty.")
        texts = [c["text"] for c in choices]
        n = len(texts)
        if n < 2:
            return [1.0] * n
        span = score_span(texts)
        byte = _mean_scores([t.encode("utf-8") for t in texts], normalized_byte_distance)
        base = torch.tensor([s + 1e-3 * b for s, b in zip(span, byte)])

        frames, grids = self._encode(images)  # temporal order
        sig = F.normalize(frames[:, -SIGLIP_DIM:], dim=-1)
        if len(frames) > self.tmax:  # the head saw at most tmax frames in training
            keep = torch.linspace(0, len(frames) - 1, self.tmax).round().long()
            frames, grids = frames[keep], grids[keep]
        fmask = torch.zeros(1, len(frames), dtype=torch.bool, device=self.device)
        video, vmask = self._video(frames[None], fmask, grids[None] if self.patch else None)
        cl, cm = self._clauses(texts)
        R = self.ref_video.shape[0]

        head = self.head(video.expand(n, -1, -1), vmask.expand(n, -1), cl, cm)
        head_prior = torch.stack([
            self.head(self.ref_video, self.ref_vmask, cl[i:i + 1].expand(R, -1, -1),
                      cm[i:i + 1].expand(R, -1)).mean() for i in range(n)])
        align, align_prior = [], []
        for i in range(n):
            c = cl[i, ~cm[i]]
            align.append(batched_align((c @ sig.T)[None])[0])
            ref = torch.einsum("kd,rtd->rkt", c, self.ref_sig).masked_fill(self.ref_mask[:, None, :], -1e4)
            align_prior.append(batched_align(ref).mean())
        align, align_prior = torch.stack(align), torch.stack(align_prior)

        final = (base + self.l1 * zscore(align - align_prior).float().cpu()
                 + self.l2 * zscore(head - head_prior).float().cpu())
        if not torch.isfinite(final).all():
            raise ValueError("Invalid scores.")
        return final.tolist()

    def predict(self, images, choices):
        scores = self.score(images, choices)
        order = sorted(range(len(choices)), key=lambda i: -scores[i])
        return [choices[i]["id"] for i in order]
