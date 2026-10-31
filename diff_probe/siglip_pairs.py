"""SigLIP zero-shot as a pairwise judge: GT vs one Qwen single-edit negative.

Centrality already narrows the choices to 2-3 that differ by one edit, so a video judge only
has to answer "which of these two matches the video". This measures how often SigLIP (no
training) gets that right, per edit kind, with the real video and with another episode's video
(if accuracy does not drop, the judge is reading the text, not the video).

Pairs: validation episodes of diff_probe/valsets_qwen.json {full,short}/A_single (GT + 4 Qwen
single edits). Frames: SigLIP image features cached in baseline_v2/cache/frames.pt.

Judges (higher score = judged to match the video):
  full        whole description vs frames, mean over frames (SigLIP truncates at 64 tokens)
  phrase_max  the edited words +/- 4 words of context, max over frames
  phrase_mean same phrase, mean over frames
  clause_max  clauses that differ between the two texts, mean of max-over-frames
  align       every clause matched to a frame in order (monotonic), mean similarity

    python diff_probe/siglip_pairs.py --out diff_probe/runs/siglip_pairs.json
"""

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline_v2"))
from model_v2 import SIGLIP_DIM, load_siglip_tokenizer, split_clauses  # noqa: E402

CONTEXT = 4
JUDGES = ["full", "phrase_max", "phrase_mean", "clause_max", "align"]


def load_pairs(path):
    sets = json.loads(Path(path).read_text())["sets"]
    pairs = []
    for view in ("full", "short"):
        for row in sets[f"{view}/A_single"]:
            gt = row["choices"][0]["text"]
            for c in row["choices"][1:]:
                pairs.append({"view": view, "episode": row["episode"], "kind": c["type"],
                              "gt": gt, "neg": c["text"]})
    return pairs


def edit_phrases(a, b):
    """The edited region of a and of b (all non-equal opcodes), widened by CONTEXT words."""
    wa, wb = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
           if o[0] != "equal"]
    i1, i2 = min(o[1] for o in ops), max(o[2] for o in ops)
    j1, j2 = min(o[3] for o in ops), max(o[4] for o in ops)
    return (" ".join(wa[max(0, i1 - CONTEXT):i2 + CONTEXT]),
            " ".join(wb[max(0, j1 - CONTEXT):j2 + CONTEXT]))


def monotonic_align(sim):
    """sim [K clauses, T frames]; best mean similarity with frame index non-decreasing in k."""
    best = sim[0].clone()
    for k in range(1, sim.shape[0]):
        best = torch.cummax(best, dim=0).values + sim[k]
    return best.max().item() / sim.shape[0]


class SiglipText:
    def __init__(self, model_dir, device):
        self.device = device
        self.tokenizer = load_siglip_tokenizer(model_dir)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = model.to(device, dtype).eval()
        self.cache = {}

    @torch.inference_mode()
    def encode(self, texts):
        todo = sorted({t for t in texts if t not in self.cache})
        for s in range(0, len(todo), 256):
            batch = todo[s:s + 256]
            ids = self.tokenizer([t.lower() for t in batch], padding="max_length", truncation=True,
                                 max_length=64, return_tensors="pt").to(self.device)
            emb = F.normalize(self.model.get_text_features(**ids).float(), dim=-1).cpu()
            self.cache.update(zip(batch, emb))
        return torch.stack([self.cache[t] for t in texts])


def judge_scores(text_enc, frames, a, b):
    """{judge: (score_a, score_b)} for one pair against one episode's frames [T, 768]."""
    out = {}
    full = text_enc.encode([a, b]) @ frames.T
    out["full"] = tuple(full.mean(1).tolist())
    pa, pb = edit_phrases(a, b)
    ph = text_enc.encode([pa, pb]) @ frames.T
    out["phrase_max"] = tuple(ph.max(1).values.tolist())
    out["phrase_mean"] = tuple(ph.mean(1).tolist())
    ca, cb = split_clauses(a), split_clauses(b)
    da, db = [c for c in ca if c not in cb] or ca, [c for c in cb if c not in ca] or cb
    out["clause_max"] = tuple((text_enc.encode(d) @ frames.T).max(1).values.mean().item()
                              for d in (da, db))
    out["align"] = tuple(monotonic_align(text_enc.encode(c) @ frames.T) for c in (ca, cb))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valsets", type=Path, default=ROOT / "diff_probe/valsets_qwen.json")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--siglip", type=Path, default=ROOT / "baseline_v2/models/siglip")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    pairs = load_pairs(args.valsets)
    cache = torch.load(args.frames, map_location="cpu", weights_only=False)
    index = {e: i for i, e in enumerate(cache["episode_ids"])}
    offsets = cache["frame_offsets"]

    def frames_of(ep):
        i = index[ep]
        sig = cache["frames"][offsets[i]:offsets[i + 1], -SIGLIP_DIM:].float()
        return F.normalize(sig, dim=-1)

    # Another episode's video: the next validation episode in a fixed order (never itself).
    episodes = sorted({p["episode"] for p in pairs}, key=lambda e: int(e.split("_")[1]))
    other = {e: episodes[(n + 1) % len(episodes)] for n, e in enumerate(episodes)}

    text_enc = SiglipText(args.siglip, args.device)
    hits = defaultdict(lambda: defaultdict(list))  # (video, view, kind) -> judge -> [0/0.5/1]
    for n, p in enumerate(pairs):
        for video, ep in (("real", p["episode"]), ("other", other[p["episode"]])):
            for judge, (sa, sb) in judge_scores(text_enc, frames_of(ep), p["gt"], p["neg"]).items():
                hit = 1.0 if sa > sb else 0.5 if sa == sb else 0.0
                for kind in (p["kind"], "all"):
                    hits[(video, p["view"], kind)][judge].append(hit)
        if n % 500 == 0:
            print(f"{n}/{len(pairs)}", flush=True)

    result = {}
    for (video, view, kind), by_judge in sorted(hits.items()):
        result[f"{video}/{view}/{kind}"] = {j: round(sum(v) / len(v), 4) for j, v in by_judge.items()}
        result[f"{video}/{view}/{kind}"]["n"] = len(by_judge["full"])
    for view in ("full", "short"):
        print(f"\n[{view}] pairwise accuracy, real video (other-episode video)")
        print(f"{'kind':16s} {'n':>4s} " + " ".join(f"{j:>18s}" for j in JUDGES))
        kinds = sorted({k.split("/")[2] for k in result if k.startswith(f"real/{view}/")},
                       key=lambda k: (k == "all", k))
        for kind in kinds:
            r, o = result[f"real/{view}/{kind}"], result[f"other/{view}/{kind}"]
            print(f"{kind:16s} {r['n']:4d} " + " ".join(f"{r[j]:.3f} ({o[j]:.3f})".rjust(18) for j in JUDGES))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
