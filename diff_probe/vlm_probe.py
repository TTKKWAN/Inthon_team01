"""Ask SmolVLM-256M (zero-shot) which reading of each differing place matches the frames.

For every episode of a local 5-choice set:
  1. spots.find_spots cuts the places where the choices differ (rule-based)
  2. per spot, SmolVLM gets the frames and a multiple-choice prompt over the variants;
     the next-token probabilities of the letters are read off (no generation)
  3. every cyclic rotation of the variant order is asked, which measures position bias
     and gives a debiased score (mean log-prob over rotations)
  4. a choice's VLM score = sum over spots of log p(its variant); ranking ties are
     broken by span centrality. "+cent" adds the centrality score on top.

Controls: frames="none" (text only) and frames="reversed" (same frames, reversed order).
A model that reads the frames should beat "none", and on direction/order edits it should
drop with "reversed".

    python diff_probe/vlm_probe.py --model models/smolvlm --sets short/H1_disjoint \
        --limit 300 --out diff_probe/results/vlm_probe.json
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.idefics3.processing_idefics3 import get_image_prompt_string

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spots import find_spots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]
LETTERS = "ABCDE"


def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def load_frames(data_dir, episode, n):
    paths = sorted((p for p in (data_dir / episode / "images").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")), key=natural_key)
    if len(paths) > n:
        paths = [paths[round(i * (len(paths) - 1) / (n - 1))] for i in range(n)]
    return [Image.open(p).convert("RGB") for p in paths]


class Judge:
    def __init__(self, model_dir, device, dtype):
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.proc.image_processor.do_image_splitting = False  # 64 tokens per frame
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_dir, dtype=dtype).to(device).eval()
        self.device = device
        tok = self.proc.tokenizer
        self.tok = tok
        self.letter_ids = []
        for letter in LETTERS:
            ids = {tok.encode(s, add_special_tokens=False)[-1] for s in (letter, " " + letter)}
            self.letter_ids.append(sorted(ids))
        self.calls = 0
        self.seconds = 0.0
        self.letter_mass = []  # prob. on the offered letters: does it answer in format?
        # Frames are preprocessed once per episode and reused by every prompt; the text
        # side gets the same image-token expansion the processor would insert.
        p = self.proc
        self.img_str = get_image_prompt_string(0, 0, p.image_seq_len, p.fake_image_token,
                                               p.image_token, p.global_image_tag)
        self.cache = {}

    def encode(self, data_dir, episode, n_frames, frames_mode):
        """Preprocessed frames on the device (None for frames_mode='none')."""
        if frames_mode == "none":
            return None
        key = (episode, n_frames)
        if key not in self.cache:
            frames = load_frames(data_dir, episode, n_frames)
            out = self.proc.image_processor([frames], return_tensors="pt",
                                            do_image_splitting=False)
            self.cache[key] = {
                "n": len(frames),
                "pv": out["pixel_values"].to(self.device, self.model.dtype),
                "pam": out["pixel_attention_mask"].to(self.device)
                if "pixel_attention_mask" in out else None}
        enc = self.cache[key]
        if frames_mode == "reversed":
            enc = {"n": enc["n"], "pv": enc["pv"].flip(1),
                   "pam": enc["pam"].flip(1) if enc["pam"] is not None else None}
        return enc

    def inputs(self, prompts, enc, padding_side):
        self.tok.padding_side = padding_side
        texts = [t.replace(self.proc.image_token, self.img_str) for t in prompts]
        batch = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        if enc is not None:
            b = len(texts)
            batch["pixel_values"] = enc["pv"].expand(b, *enc["pv"].shape[1:]).contiguous()
            if enc["pam"] is not None:
                batch["pixel_attention_mask"] = enc["pam"].expand(
                    b, *enc["pam"].shape[1:]).contiguous()
        return batch

    def prompt(self, n_frames, variants):
        lines = [f"{LETTERS[i]}. {v}" for i, v in enumerate(variants)]
        head = (f"These are {n_frames} frames from one video, in time order.\n"
                if n_frames else "")
        text = (head + "Which phrase correctly describes what happens in the video?\n"
                + "\n".join(lines) + "\nAnswer with the letter only.")
        content = [{"type": "image"} for _ in range(n_frames)] + [{"type": "text", "text": text}]
        return self.proc.apply_chat_template([{"role": "user", "content": content}],
                                             add_generation_prompt=True)

    @torch.no_grad()
    def letter_logprobs(self, enc, variant_lists):
        """One forward for a batch of prompts that share the frames."""
        prompts = [self.prompt(enc["n"] if enc else 0, v) for v in variant_lists]
        batch = self.inputs(prompts, enc, "left")
        t0 = time.time()
        logits = self.model(**batch).logits[:, -1].float()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.seconds += time.time() - t0
        self.calls += 1
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        for row, variants in zip(logp, variant_lists):
            scores = torch.stack([torch.logsumexp(row[self.letter_ids[i]], 0)
                                  for i in range(len(variants))])
            self.letter_mass.append(torch.logsumexp(scores, 0).exp().item())
            out.append(torch.log_softmax(scores, 0).tolist())
        return out


@torch.no_grad()
def text_logprob(judge, enc, texts):
    """Sum log p(text | frames) as the assistant's answer to a describe prompt."""
    content = [{"type": "image"} for _ in range(enc["n"] if enc else 0)] + [
        {"type": "text", "text": "Describe what happens in the video."}]
    user = [{"role": "user", "content": content}]
    prompt = judge.proc.apply_chat_template(user, add_generation_prompt=True)
    fulls = [prompt + " " + t + "<end_of_utterance>" for t in texts]
    batch = judge.inputs([prompt] + fulls, enc, "right")
    start = int(batch["attention_mask"][0].sum())  # prompt length (same prefix for all)
    t0 = time.time()
    logits = judge.model(**batch).logits.float()
    if judge.device.startswith("cuda"):
        torch.cuda.synchronize()
    judge.seconds += time.time() - t0
    judge.calls += 1
    logp = torch.log_softmax(logits[1:, :-1], dim=-1)
    target = batch["input_ids"][1:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = batch["attention_mask"][1:, 1:].clone()
    mask[:, :start - 1] = 0
    return (tok_lp * mask).sum(-1).tolist()


def judge_spot_pmi(judge, enc, variants):
    """log p(v | frames) - log p(v | no frames): image evidence with the text prior removed.
    No letters, so no position bias."""
    with_frames = text_logprob(judge, enc, variants) if enc else [0.0] * len(variants)
    without = text_logprob(judge, None, variants)
    pmi = [a - b for a, b in zip(with_frames, without)] if enc else without
    z = math.log(sum(math.exp(x - max(pmi)) for x in pmi)) + max(pmi)
    scores = [x - z for x in pmi]
    return scores, scores, []


def judge_spot(judge, enc, variants):
    """Return debiased log-probs per variant, first-order log-probs, chosen positions."""
    k = len(variants)
    rotations = [[(r + i) % k for i in range(k)] for r in range(k)]
    results = judge.letter_logprobs(enc, [[variants[j] for j in rot] for rot in rotations])
    total = [0.0] * k
    picked_positions = []
    for rot, lp in zip(rotations, results):
        for pos, j in enumerate(rot):
            total[j] += lp[pos]
        picked_positions.append(max(range(k), key=lambda p: lp[p]))
    debiased = [t / k for t in total]
    z = math.log(sum(math.exp(x) for x in debiased))
    first = results[0]  # rotation 0 = original order
    return [x - z for x in debiased], first, picked_positions


def rank_score(scores, pos_index):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return RANK_SCORE[order.index(pos_index)]


def run(judge, rows, data_dir, n_frames, frames_mode, mode, context, method="mc", seed=0):
    spot_fn = judge_spot_pmi if method == "pmi" else judge_spot
    rng = random.Random(seed)
    stats = {"spots": 0, "spot_correct": 0, "spot_correct_first": 0, "spot_chance": 0.0,
             "pos_hist": {}, "vlm": 0.0, "cent": 0.0, "vlm_cent": 0.0, "episodes": 0}
    for row in rows:
        choices = list(row["choices"])
        rng.shuffle(choices)
        texts = [c["text"] for c in choices]
        pos_index = next(i for i, c in enumerate(choices) if c["type"] == "pos")
        found = find_spots(texts, context=context, mode=mode)
        cent = found["centrality"]
        frames = judge.encode(data_dir, row["episode"], n_frames, frames_mode)
        vlm = [0.0] * len(texts)
        for spot in found["spots"]:
            variants, cv = spot["variants"], spot["choice_variant"]
            if len(variants) > len(LETTERS):
                continue
            debiased, first, picked = spot_fn(judge, frames, variants)
            truth = cv[pos_index]
            stats["spots"] += 1
            stats["spot_correct"] += max(range(len(variants)), key=lambda j: debiased[j]) == truth
            stats["spot_correct_first"] += max(range(len(variants)), key=lambda j: first[j]) == truth
            stats["spot_chance"] += 1 / len(variants)
            if picked:
                hist = stats["pos_hist"].setdefault(str(len(variants)), [0] * len(variants))
                for p in picked:
                    hist[p] += 1
            for i in range(len(texts)):
                vlm[i] += debiased[cv[i]]
        # tiny centrality term only breaks VLM ties
        stats["vlm"] += rank_score([v + 1e-6 * c for v, c in zip(vlm, cent)], pos_index)
        stats["cent"] += rank_score(cent, pos_index)
        stats["vlm_cent"] += rank_score([v + c for v, c in zip(vlm, cent)], pos_index)
        stats["episodes"] += 1
    n, s = stats["episodes"], max(stats["spots"], 1)
    return {"episodes": n, "spots": stats["spots"],
            "spot_acc": stats["spot_correct"] / s,
            "spot_acc_first_order": stats["spot_correct_first"] / s,
            "spot_chance": stats["spot_chance"] / s,
            "position_hist": {k: [round(x / sum(v), 3) for x in v]
                              for k, v in stats["pos_hist"].items()},
            "score_vlm": stats["vlm"] / n, "score_centrality": stats["cent"] / n,
            "score_vlm_plus_centrality": stats["vlm_cent"] / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "diff_probe/models/smolvlm"))
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "Train" if (ROOT / "Train").exists() else ROOT / "data/Train")
    ap.add_argument("--valsets", type=Path, default=ROOT / "diff_probe/valsets.json")
    ap.add_argument("--sets", nargs="+", default=["short/H1_disjoint", "short/H2_samespot",
                                                  "short/H6_fact_a_b_ab_c", "short/H7_fact_a_b_ab_ac"])
    ap.add_argument("--configs", nargs="+", default=["atoms:3", "merged:4", "merged:full"])
    ap.add_argument("--methods", nargs="+", default=["mc", "pmi"])
    ap.add_argument("--frames", nargs="+", default=["normal", "reversed", "none"])
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "diff_probe/results/vlm_probe.json")
    args = ap.parse_args()

    sets = json.loads(args.valsets.read_text())["sets"]
    judge = Judge(args.model, args.device, torch.float16)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for set_name in args.sets:
        rows = sets[set_name][:args.limit]
        for cfg in args.configs:
            mode, ctx = cfg.split(":")
            ctx = ctx if ctx in ("full", "clause") else int(ctx)
            for method in args.methods:
              for fm in args.frames:
                if method == "pmi" and fm == "none":
                    continue  # pmi without frames is just the text prior
                c0, t0, w0 = judge.calls, judge.seconds, time.time()
                m0 = len(judge.letter_mass)
                r = run(judge, rows, args.data_dir, args.n_frames, fm, mode, ctx, method)
                mass = judge.letter_mass[m0:]
                r.update({"set": set_name, "config": cfg, "method": method, "frames": fm,
                          "letter_mass": round(sum(mass) / max(len(mass), 1), 3),
                          "forward_calls": judge.calls - c0,
                          "forward_seconds": round(judge.seconds - t0, 1),
                          "wall_seconds": round(time.time() - w0, 1)})
                results.append(r)
                print(json.dumps(r), flush=True)
                args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
