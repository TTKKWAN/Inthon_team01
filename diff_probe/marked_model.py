"""The trained head of the marked-token verifier (shared by training and the submission).

Video: per-frame features [DINOv2 CLS ; DINOv2 mean patch ; SigLIP] -> linear + frame-order
embedding -> transformer. Text: frozen MPNet token states + a learned mark embedding on
the words where the choices differ; tokens cross-attend to the frames, and the per-token
scores are averaged over the marked tokens only.
"""

import torch
from torch import nn


class MarkedVerifier(nn.Module):
    def __init__(self, frame_dim=2304, text_dim=768, d=256, layers=2, heads=4,
                 max_frames=32, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.frame_pos = nn.Embedding(max_frames, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.frame_enc = nn.TransformerEncoder(enc, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.mark_emb = nn.Embedding(2, d)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.text_dec = nn.TransformerDecoder(dec, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def encode_video(self, frames, mask):
        t = torch.arange(frames.shape[1], device=frames.device)
        x = self.frame_proj(frames) + self.frame_pos(t)[None]
        return self.frame_enc(x, src_key_padding_mask=~mask)

    def score(self, text_h, text_mask, marks, mem, mem_mask):
        x = self.text_proj(text_h) + self.mark_emb(marks)
        h = self.text_dec(x, mem, tgt_key_padding_mask=~text_mask, memory_key_padding_mask=~mem_mask)
        s = self.out(h).squeeze(-1)
        w = marks.float() * text_mask.float()
        empty = w.sum(1) == 0
        w[empty] = text_mask[empty].float()  # no marked token: fall back to all tokens
        return (s * w).sum(1) / w.sum(1)
