"""Video judges shipped with the submission (copied from diff_probe/pair_head.py and
diff_probe/rerank_eval.py so the submission folder is self-contained)."""

import torch
from torch import nn

MAX_CLAUSES = 8
GRID_TOKENS = 16


class PairHead(nn.Module):
    def __init__(self, frame_dim=2304, text_dim=768, d=256, layers=2, heads=4, tmax=64, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.frame_pos = nn.Parameter(torch.zeros(tmax, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.clause_pos = nn.Parameter(torch.zeros(MAX_CLAUSES, d))
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.query = nn.Linear(d, d)

    def encode_video(self, frames, fmask):
        h = self.frame_proj(frames) + self.frame_pos[:frames.shape[1]]
        return self.temporal(h, src_key_padding_mask=fmask)

    def forward(self, video, fmask, clauses, cmask):
        q = self.text_proj(clauses) + self.clause_pos[:clauses.shape[1]]
        summary, _ = self.cross(q, video, video, key_padding_mask=fmask)
        per = (self.query(q) * summary).sum(-1) / q.shape[-1] ** 0.5
        keep = (~cmask).float()
        return (per * keep).sum(1) / keep.sum(1).clamp_min(1)


class PatchHead(nn.Module):
    """diff_probe/patch_head.py: per frame 1 global token + 16 DINOv2 grid tokens (+ change from
    the previous frame), transformer over all of them, clauses cross-attend."""

    def __init__(self, frame_dim=2304, patch_dim=768, text_dim=768, d=256, layers=2, heads=4,
                 tmax=64, dropout=0.1):
        super().__init__()
        self.frame_proj = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, d))
        self.patch_proj = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d))
        self.delta_proj = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, d))
        self.frame_pos = nn.Parameter(torch.zeros(tmax, d))
        self.slot_pos = nn.Parameter(torch.zeros(1 + GRID_TOKENS, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, d))
        self.clause_pos = nn.Parameter(torch.zeros(MAX_CLAUSES, d))
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.query = nn.Linear(d, d)

    def encode_video(self, frames, fmask, patches):
        B, T = fmask.shape
        delta = patches - torch.cat([patches[:, :1], patches[:, :-1]], 1)
        grid = self.patch_proj(patches) + self.delta_proj(delta)
        tokens = torch.cat([self.frame_proj(frames)[:, :, None], grid], 2)
        tokens = tokens + self.frame_pos[:T, None] + self.slot_pos[None, None]
        tmask = fmask[:, :, None].expand(B, T, 1 + GRID_TOKENS).reshape(B, -1)
        h = self.encoder(tokens.reshape(B, T * (1 + GRID_TOKENS), -1), src_key_padding_mask=tmask)
        return h, tmask

    def forward(self, video, vmask, clauses, cmask):
        q = self.text_proj(clauses) + self.clause_pos[:clauses.shape[1]]
        summary, _ = self.cross(q, video, video, key_padding_mask=vmask)
        per = (self.query(q) * summary).sum(-1) / q.shape[-1] ** 0.5
        keep = (~cmask).float()
        return (per * keep).sum(1) / keep.sum(1).clamp_min(1)


def batched_align(sim):
    """sim [R, K, T] -> [R] monotonic clause-to-frame alignment, mean over clauses."""
    best = sim[:, 0]
    for k in range(1, sim.shape[1]):
        best = torch.cummax(best, dim=1).values + sim[:, k]
    return best.max(1).values / sim.shape[1]
