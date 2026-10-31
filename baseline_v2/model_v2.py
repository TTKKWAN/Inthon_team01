"""Order-aware video/text matching on top of frozen allowed encoders.

Shared by training and the exported submission, so both use one implementation.

Video : every frame -> [DINOv2 CLS ; DINOv2 mean(patch) ; SigLIP image]   (2304-d)
Text  : description split into ordered clauses ->
        clause = [MPNet mean-pool ; SigLIP text]                            (1536-d)
        full   = [MPNet(whole description) ; mean SigLIP over clauses]      (1536-d)
Scorer: small joint transformer over [CLS, frames, full text, clauses].
        Each clause token also gets an explicit alignment signal from the
        SigLIP clause x frame similarity: where in time it matches (soft argmax)
        and how strongly (max sim). Clause tokens carry their order index, so
        the model can tell "A then B" from "B then A".
"""

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, PreTrainedTokenizerFast

IMAGE_MODEL_ID = "facebook/dinov2-base"
TEXT_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
MAX_FRAMES = 32
MAX_CLAUSES = 8
SIGLIP_DIM = 768  # SigLIP part is always the LAST 768 dims of frame / clause features.

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(
    r",\s*(?:and\s+)?then\s+|;\s*|\s+and\s+then\s+|,\s*(?:before|after which|followed by)\s+",
    re.IGNORECASE,
)


def split_clauses(text, max_clauses=MAX_CLAUSES):
    """Sentence + 'then'-style split. Keeps event order; merges the tail if too long."""
    parts = []
    for sentence in _SENTENCE.split(text.strip()):
        parts.extend(p.strip(" ,.") for p in _CLAUSE.split(sentence))
    parts = [p for p in parts if p] or [text.strip()]
    if len(parts) > max_clauses:
        parts = parts[:max_clauses - 1] + [" ".join(parts[max_clauses - 1:])]
    return parts


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class OrderAwareScorer(nn.Module):
    def __init__(self, frame_dim=2304, text_dim=1536, d_model=256, layers=2,
                 heads=4, dropout=0.1, max_clauses=MAX_CLAUSES):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d_model))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d_model))
        self.align_proj = nn.Linear(3, d_model)
        self.summary_proj = nn.Linear(4, d_model)
        self.align_scale = nn.Parameter(torch.tensor(20.0))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.frame_pos = nn.Embedding(MAX_FRAMES, d_model)
        self.clause_pos = nn.Embedding(max_clauses, d_model)
        # 0 = cls, 1 = frame, 2 = full text, 3 = clause
        self.type_emb = nn.Embedding(4, d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    @staticmethod
    def frame_positions(lengths, max_len, device):
        """Map each episode's frames onto MAX_FRAMES slots by relative time."""
        idx = torch.arange(max_len, device=device).unsqueeze(0).float()
        span = (lengths.float() - 1).clamp_min(1).unsqueeze(1)
        return (idx / span * (MAX_FRAMES - 1)).round().long().clamp(0, MAX_FRAMES - 1)

    def alignment(self, frames, frame_mask, clauses, lengths):
        """[B,K,3]: soft time position in [0,1], max sim, and position rank gap to previous."""
        v = F.normalize(frames[..., -SIGLIP_DIM:], dim=-1)
        c = F.normalize(clauses[..., -SIGLIP_DIM:], dim=-1)
        sim = torch.einsum("bkd,btd->bkt", c, v)
        sim = sim.masked_fill(~frame_mask.unsqueeze(1), -1e4)
        attn = (sim * self.align_scale).softmax(-1)
        rel = torch.arange(frames.shape[1], device=frames.device).float().unsqueeze(0)
        rel = rel / (lengths.float() - 1).clamp_min(1).unsqueeze(1)
        pos = torch.einsum("bkt,bt->bk", attn, rel)
        gap = pos - torch.cat([torch.zeros_like(pos[:, :1]), pos[:, :-1]], dim=1)
        return torch.stack([pos, sim.max(-1).values, gap], dim=-1)

    @staticmethod
    def grounding_summary(align, clause_mask):
        """[B,4]: weakest clause match, mean match, backward time jumps, clause count.
        An appended / substituted event shows up as a low minimum even when every
        other clause matches well."""
        m = clause_mask.float()
        best = align[..., 1]
        weakest = best.masked_fill(~clause_mask, 1e4).min(1).values
        mean = (best * m).sum(1) / m.sum(1).clamp_min(1)
        backward = (torch.relu(-align[..., 2]) * m).sum(1)
        return torch.stack([weakest, mean, backward, m.sum(1) / MAX_CLAUSES], dim=-1)

    def forward(self, frames, frame_mask, full_text, clauses, clause_mask):
        """frames [B,T,F], frame_mask [B,T] (True=valid), full_text [B,D],
        clauses [B,K,D], clause_mask [B,K]. Returns logits [B]."""
        b, t, _ = frames.shape
        k = clauses.shape[1]
        device = frames.device
        types = self.type_emb.weight
        lengths = frame_mask.sum(1)
        pos = self.frame_positions(lengths, t, device)
        v = self.frame_proj(frames) + self.frame_pos(pos) + types[1]
        full = self.text_proj(full_text).unsqueeze(1) + types[2]
        align = self.alignment(frames, frame_mask, clauses, lengths)
        c = (self.text_proj(clauses) + types[3]
             + self.clause_pos(torch.arange(k, device=device)).unsqueeze(0)
             + self.align_proj(align))
        cls = (self.cls.expand(b, -1, -1) + types[0]
               + self.summary_proj(self.grounding_summary(align, clause_mask)).unsqueeze(1))
        tokens = torch.cat([cls, v, full, c], dim=1)
        one = torch.ones(b, 1, dtype=torch.bool, device=device)
        valid = torch.cat([one, frame_mask, one, clause_mask], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=~valid)
        return self.head(out[:, 0]).squeeze(-1)


def _load(model_dir, dtype=torch.float32):
    return AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=dtype)


def load_siglip_tokenizer(model_dir):
    """SigLIP's default SiglipTokenizer needs sentencepiece, which is not in the grading
    environment. Its tokenizer.json gives the same ids through `tokenizers` alone.
    Only input_ids are returned: SigLIP was trained without an attention mask."""
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"SigLIP tokenizer.json missing (rerun download_siglip.py): {path}")
    return PreTrainedTokenizerFast(tokenizer_file=str(path), eos_token="</s>", pad_token="</s>",
                                   unk_token="<unk>", model_max_length=64,
                                   model_input_names=["input_ids"])


class FrozenEncoders:
    def __init__(self, image_model_dir, text_model_dir, siglip_model_dir, device="cpu",
                 image_batch_size=32, text_batch_size=64, clause_lexicon=None,
                 max_clauses=MAX_CLAUSES):
        self.device = torch.device(device)
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        # clause_lexicon: verb lexicon JSON for the clause_split event segmenter; without it
        # the original split_clauses rules are used.
        if clause_lexicon:
            from clause_split import Segmenter
            self.split = Segmenter.from_json(clause_lexicon, max_pieces=max_clauses).texts
        else:
            self.split = lambda text: split_clauses(text, max_clauses)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            image_model_dir, local_files_only=True, use_fast=False)
        self.siglip_processor = AutoImageProcessor.from_pretrained(
            siglip_model_dir, local_files_only=True, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_dir, local_files_only=True)
        self.siglip_tokenizer = load_siglip_tokenizer(siglip_model_dir)
        self.image_model = _load(image_model_dir)
        self.text_model = _load(text_model_dir)
        self.siglip = _load(siglip_model_dir)
        for model, kind, model_id in [(self.image_model, "dinov2", IMAGE_MODEL_ID),
                                      (self.text_model, "mpnet", TEXT_MODEL_ID),
                                      (self.siglip, "siglip", SIGLIP_MODEL_ID)]:
            if model.config.model_type != kind:
                raise ValueError(f"Use the allowed model: {model_id}")
        self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        for model in (self.image_model, self.text_model, self.siglip):
            model.requires_grad_(False).to(device=self.device, dtype=self.dtype).eval()

    def preprocess(self, images):
        images = [im.convert("RGB") for im in images]
        return (self.dino_processor(images=images, return_tensors="pt")["pixel_values"],
                self.siglip_processor(images=images, return_tensors="pt")["pixel_values"])

    @torch.inference_mode()
    def encode_pixels(self, dino_px, siglip_px):
        hidden = self.image_model(
            pixel_values=dino_px.to(self.device, self.dtype)).last_hidden_state
        sig = self.siglip.get_image_features(pixel_values=siglip_px.to(self.device, self.dtype))
        return torch.cat([hidden[:, 0], hidden[:, 1:].mean(1), F.normalize(sig, dim=-1)],
                         dim=-1).float()

    @torch.inference_mode()
    def encode_frames(self, images):
        """[T, 2304] per-frame features, in the given (temporal) order."""
        out = []
        for s in range(0, len(images), self.image_batch_size):
            out.append(self.encode_pixels(*self.preprocess(images[s:s + self.image_batch_size]))
                       .cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def encode_mpnet(self, texts):
        out = []
        for s in range(0, len(texts), self.text_batch_size):
            inputs = self.tokenizer(texts[s:s + self.text_batch_size], padding=True,
                                    truncation=True, max_length=384,
                                    return_tensors="pt").to(self.device)
            tokens = self.text_model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            out.append(((tokens * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def encode_siglip_text(self, texts):
        out = []
        for s in range(0, len(texts), self.text_batch_size):
            # SigLIP was trained with fixed-length (64) padding.
            inputs = self.siglip_tokenizer([t.lower() for t in texts[s:s + self.text_batch_size]],
                                           padding="max_length", truncation=True, max_length=64,
                                           return_tensors="pt").to(self.device)
            out.append(F.normalize(self.siglip.get_text_features(**inputs).float(), dim=-1).cpu())
        return torch.cat(out)

    def encode_descriptions(self, texts):
        """Returns (full [N,1536], clauses [N,K,1536], clause_mask [N,K])."""
        clause_lists = [self.split(t) for t in texts]
        flat = [c for cl in clause_lists for c in cl]
        clause_emb = torch.cat([self.encode_mpnet(flat), self.encode_siglip_text(flat)], dim=-1)
        full_mpnet = self.encode_mpnet(list(texts))
        k = max(len(cl) for cl in clause_lists)
        clauses = torch.zeros(len(texts), k, clause_emb.shape[1])
        mask = torch.zeros(len(texts), k, dtype=torch.bool)
        full_sig = torch.zeros(len(texts), SIGLIP_DIM)
        i = 0
        for n, cl in enumerate(clause_lists):
            clauses[n, :len(cl)] = clause_emb[i:i + len(cl)]
            mask[n, :len(cl)] = True
            full_sig[n] = F.normalize(clause_emb[i:i + len(cl), -SIGLIP_DIM:].mean(0), dim=-1)
            i += len(cl)
        return torch.cat([full_mpnet, full_sig], dim=-1), clauses, mask

    def export(self, destination):
        destination = Path(destination)
        for name, model in [("dinov2", self.image_model), ("mpnet", self.text_model),
                            ("siglip", self.siglip)]:
            state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                     for k, v in model.state_dict().items()}
            model.save_pretrained(destination / name, state_dict=state, safe_serialization=True)
        self.dino_processor.save_pretrained(destination / "dinov2")
        self.tokenizer.save_pretrained(destination / "mpnet")
        self.siglip_processor.save_pretrained(destination / "siglip")
        self.siglip_tokenizer.save_pretrained(destination / "siglip")
