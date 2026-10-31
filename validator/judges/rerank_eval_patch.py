"""rerank_eval.py with the patch_head.py judge in place of pair_head.py (same arguments).

    python validator/judges/rerank_eval_patch.py --head diff_probe/runs/patch_head/epochNN.pt \
        --hf diff_probe/runs/qwen_negatives_from_gt.json --span-dir diff_probe/runs/span_sub
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diff_probe"))
import rerank_eval as R  # noqa: E402
from patch_head import PatchBank, PatchHead  # noqa: E402
from siglip_pairs import SIGLIP_DIM  # noqa: E402

PATCHES = ROOT / "baseline_v2/cache/patches4.pt"


class PatchJudges(R.Judges):
    def __init__(self, head_path, frames, clauses, ref_eps, device):
        ckpt = torch.load(head_path, map_location="cpu", weights_only=False)
        self.model = PatchHead(tmax=ckpt["tmax"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.frames, self.clauses, self.device, self.ref_eps = frames, clauses, device, ref_eps
        self.patches = PatchBank(PATCHES, frames)
        with torch.no_grad():
            fr, self.ref_fmask = frames.batch(ref_eps, device)
            self.ref_video, self.ref_vmask = self.model.encode_video(fr, self.ref_fmask,
                                                                     self.patches.batch(ref_eps, device))
            self.ref_sig = F.normalize(fr[..., -SIGLIP_DIM:], dim=-1)
        self.prior = {}

    @torch.no_grad()
    def _priors(self, texts):
        R_ = len(self.ref_eps)
        for t in [t for t in dict.fromkeys(texts) if t not in self.prior]:
            cl, cm = self.clauses.batch([t], self.device)
            k = int((~cm[0]).sum())
            head = self.model(self.ref_video, self.ref_vmask, cl.expand(R_, -1, -1), cm.expand(R_, -1)).mean()
            sim = torch.einsum("kd,rtd->rkt", cl[0, :k], self.ref_sig)
            sim = sim.masked_fill(self.ref_fmask[:, None, :], -1e4)
            weak = sim.max(2).values.min(1).values.mean()
            self.prior[t] = (R.batched_align(sim).mean().item(), head.item(), weak.item())

    @torch.no_grad()
    def score(self, texts, ep):
        self._priors(texts)
        fr, fm = self.frames.batch([ep], self.device)
        video, vm = self.model.encode_video(fr, fm, self.patches.batch([ep], self.device))
        cl, cm = self.clauses.batch(texts, self.device)
        n = len(texts)
        head = self.model(video.expand(n, -1, -1), vm.expand(n, -1), cl, cm).tolist()
        sig = F.normalize(fr[0, ~fm[0], -SIGLIP_DIM:], dim=-1)
        align, weak = [], []
        for i in range(n):
            k = int((~cm[i]).sum())
            sim = cl[i, :k] @ sig.T
            align.append(R.batched_align(sim[None]).item())
            weak.append(sim.max(1).values.min().item())
        return {"align_db": [a - self.prior[t][0] for a, t in zip(align, texts)],
                "head_db": [h - self.prior[t][1] for h, t in zip(head, texts)],
                "weak_db": [w - self.prior[t][2] for w, t in zip(weak, texts)],
                "align": align, "head": head}


if __name__ == "__main__":
    R.Judges = PatchJudges
    R.main()
