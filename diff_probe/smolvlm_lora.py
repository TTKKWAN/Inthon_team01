"""SmolVLM-256M + LoRA as a caption verifier: score(caption) = sum log p(caption | frames).

The vision tower and connector stay frozen, so each episode's frames are encoded once
(`cache`, at most 12 evenly spaced frames, 64 tokens each) and LoRA trains the language
model only (`train`). Per Train episode, GT and K Qwen negatives (train split only) are
scored with the episode's own frames and with another Train episode's frames:
  text loss   listwise CE over db(c) = score(c | own) - score(c | other)   (--loss db)
              Whatever the wording alone earns appears in both terms and cancels, so the
              ranking can only be learned from the frames. (--loss raw: CE over score(c | own);
              it learned Qwen's wording: 0.86 pairwise with another episode's frames.)
  video loss  CE: GT scores higher with its own frames than with the other frames
At test time the reference is a few fixed Train videos (--refs), never other test episodes.
Data: Train frames + qwen_negatives_from_gt.json (team augmentation). Nothing from the
validation split, TemporalBench or any teacher's scores is used for training.

`eval_tb` scores TemporalBench cells (external_eval/temporalbench, EVALUATION ONLY) with
own frames, another TemporalBench video (control) and --refs fixed Train videos; `report_tb`
prints pairwise accuracy by edit category and 5-choice scores of span + λ·z(vlm), halves A / B.

    python diff_probe/smolvlm_lora.py cache --episodes data/Train --out diff_probe/runs/vlm_cache/train.pt
    python diff_probe/smolvlm_lora.py train --cache diff_probe/runs/vlm_cache/train.pt \
        --hf diff_probe/runs/qwen_negatives_from_gt.json --out diff_probe/runs/vlm_lora
    python diff_probe/smolvlm_lora.py eval_tb --cache diff_probe/runs/vlm_cache/tb.pt \
        --adapter diff_probe/runs/vlm_lora/epoch1 --out diff_probe/runs/vlm_lora/tb_epoch1.json
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.idefics3.processing_idefics3 import get_image_prompt_string

ROOT = Path(__file__).resolve().parents[1]
MAX_FRAMES = 12
PROMPT = "Describe what happens in the video."
RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]
TB = ROOT / "external_eval/temporalbench"


def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def load_frames(folder):
    paths = sorted((p for p in (folder / "images").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")), key=natural_key)
    if len(paths) > MAX_FRAMES:
        paths = [paths[round(i * (len(paths) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)]
    return [Image.open(p).convert("RGB") for p in paths]


class VLM:
    def __init__(self, model_dir, device, lora_rank=0, adapter=None):
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.proc.image_processor.do_image_splitting = False  # 64 tokens per frame
        self.proc.image_processor.size = {"longest_edge": 512}
        self.tok = self.proc.tokenizer
        self.device = torch.device(device)
        # Frozen base in fp16 (fp32 weights would be re-cast by autocast in every layer, every call);
        # peft keeps the LoRA weights in fp32.
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForVision2Seq.from_pretrained(model_dir, dtype=self.dtype).to(self.device)
        model.requires_grad_(False)
        self.core, self.lm_head = model.model, model.lm_head  # LoRA is injected into these modules
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        elif lora_rank:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
        self.model = model.eval()
        p = self.proc
        self.img_str = get_image_prompt_string(0, 0, p.image_seq_len, p.fake_image_token,
                                               p.image_token, p.global_image_tag)
        self.prefix = {}

    def preprocess(self, frames):
        """CPU part (resize, normalize): pixel_values [1, n, 3, 512, 512] and its mask."""
        out = self.proc.image_processor([frames], return_tensors="pt", do_image_splitting=False)
        return out["pixel_values"], out.get("pixel_attention_mask")

    @torch.no_grad()
    def encode(self, pv, pam):
        """[n, 64, 576] connector outputs for one episode's frames (fp16)."""
        pam = pam.to(self.device) if pam is not None else None
        with torch.autocast(self.device.type, dtype=torch.float16):
            return self.core.get_image_features(pv.to(self.device), pam).half()

    def prefix_ids(self, n):
        if n not in self.prefix:
            content = [{"type": "image"} for _ in range(n)] + [{"type": "text", "text": PROMPT}]
            text = self.proc.apply_chat_template([{"role": "user", "content": content}],
                                                 add_generation_prompt=True)
            self.prefix[n] = self.tok(text.replace(self.proc.image_token, self.img_str))["input_ids"]
        return self.prefix[n]

    def score(self, feats, texts):
        """Sum log p(text | frames) per text, differentiable. feats: [n, 64, H] of one video."""
        return self.score_videos([(feats, texts)])[0]

    def score_videos(self, videos):
        """[(feats, texts)] -> one score tensor per video, all in one batched forward.
        (Sharing each video's ~780-token prompt through a KV cache was tested: same scores, but
        2x slower here, since the many small kernels cost more than the saved compute.)"""
        sizes = [len(ts) for _, ts in videos]
        return list(self.score_items([(f, t) for f, ts in videos for t in ts]).split(sizes))

    def score_items(self, items):
        """Like score, but every (feats, text) row has its own video: many episodes, one forward.
        Idefics3 fills each row's image tokens from consecutive blocks of image_hidden_states."""
        pres = [self.prefix_ids(len(f)) for f, _ in items]
        conts = [self.tok(" " + t + "<end_of_utterance>", add_special_tokens=False)["input_ids"] for _, t in items]
        L = max(len(p) + len(c) for p, c in zip(pres, conts))
        ids = torch.full((len(items), L), self.tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros(len(items), L, dtype=torch.long)
        tgt = torch.zeros(len(items), L, dtype=torch.bool)
        for i, (p, c) in enumerate(zip(pres, conts)):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
            att[i, :len(p) + len(c)] = 1
            tgt[i, len(p):len(p) + len(c)] = True
        ids, att, tgt = ids.to(self.device), att.to(self.device), tgt.to(self.device)
        img = torch.cat([f for f, _ in items]).to(self.device, self.dtype)
        with torch.autocast(self.device.type, dtype=torch.float16):
            hidden = self.core(input_ids=ids, attention_mask=att, image_hidden_states=img).last_hidden_state
            pos = tgt[:, 1:]  # hidden at t-1 predicts token t
            logits = self.lm_head(hidden[:, :-1][pos]).float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:][pos][:, None])[:, 0]
        row = torch.arange(len(items), device=self.device)[:, None].expand_as(pos)[pos]
        return torch.zeros(len(items), device=self.device).index_add(0, row, lp)


# ---------------------------------------------------------------- cache

class EpisodePixels(torch.utils.data.Dataset):
    """Preprocessing runs in DataLoader worker processes; the GPU only runs the encoder."""

    def __init__(self, vlm, root, eps):
        self.vlm, self.root, self.eps = vlm, root, eps

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        return (self.eps[i], *self.vlm.preprocess(load_frames(self.root / self.eps[i])))


def cmd_cache(args):
    vlm = VLM(args.model, args.device)
    eps = sorted(p.name for p in args.episodes.iterdir() if (p / "images").is_dir())[:args.limit]
    out, t0 = {}, time.time()
    loader = torch.utils.data.DataLoader(EpisodePixels(vlm, args.episodes, eps), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=4)
    for i, (ep, pv, pam) in enumerate(loader):
        out[ep] = vlm.encode(pv, pam).cpu()
        if i % 500 == 0:
            print(f"{i}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"{len(out)} episodes -> {args.out} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- train

def load_groups(hf_path, splits_path, part):
    episodes = json.loads(Path(hf_path).read_text())["episodes"]
    eps = json.loads(Path(splits_path).read_text())[part]
    groups = [{"episode": e, "view": "full" if i == 0 else "short", "pos": g["pos"],
               "negs": [(k, t) for k, t in g["negs"] if t != g["pos"]]}
              for e in eps for i, g in enumerate(episodes.get(e, []))]
    return [g for g in groups if g["negs"]]


@torch.no_grad()
def pairwise_valid(vlm, groups, feats, n):
    """In-domain check on the validation split: P(GT > neg) with own frames, another episode's
    frames, and the debiased difference own - other."""
    vlm.model.eval()
    hits = defaultdict(list)
    eps = sorted({g["episode"] for g in groups})
    other = {e: eps[(i + 1) % len(eps)] for i, e in enumerate(eps)}
    for g in groups[:n]:
        texts = [g["pos"]] + [t for _, t in g["negs"]]
        own, oth = vlm.score_videos([(feats[g["episode"]], texts), (feats[other[g["episode"]]], texts)])
        for video, s in (("own", own), ("other", oth), ("db", own - oth)):
            s = s.tolist()
            for (k, _), sn in zip(g["negs"], s[1:]):
                h = 1.0 if s[0] > sn else 0.5 if s[0] == sn else 0.0
                hits[f"{video}/{k}"].append(h)
                hits[f"{video}/all"].append(h)
    vlm.model.train()
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(hits.items())}


def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    vlm = VLM(args.model, args.device, lora_rank=args.rank)
    feats = torch.load(args.cache)
    train = [g for g in load_groups(args.hf, args.splits, "train") if g["episode"] in feats]
    valid = [g for g in load_groups(args.hf, args.splits, "valid") if g["episode"] in feats]
    rng.shuffle(valid)
    train_eps = sorted({g["episode"] for g in train})
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    print(f"train groups {len(train)}, valid groups {len(valid)}, trainable {sum(p.numel() for p in params):,}",
          flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * (len(train[:args.limit]) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(total, 1)))
    scaler = torch.amp.GradScaler()
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")
    vlm.model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = train[:args.limit]
        rng.shuffle(order)
        run = defaultdict(float)
        for b in range(0, len(order) - args.batch + 1, args.batch):
            # One optimizer step = args.batch episodes, scored in forwards of args.micro episodes.
            for m in range(b, b + args.batch, args.micro):
                videos = []
                for g in order[m:m + args.micro]:
                    negs = rng.sample(g["negs"], min(args.k, len(g["negs"])))
                    other = rng.choice(train_eps)
                    while other == g["episode"]:
                        other = rng.choice(train_eps)
                    texts = [g["pos"]] + [t for _, t in negs]
                    videos += [(feats[g["episode"]], texts), (feats[other], texts)]
                s = vlm.score_videos(videos)
                own, oth = s[0::2], s[1::2]
                spans = own
                zero = torch.zeros(1, dtype=torch.long, device=own[0].device)
                rank = [o - r for o, r in zip(own, oth)] if args.loss == "db" else own
                loss_text = sum(F.cross_entropy(x[None], zero) for x in rank) / len(spans)
                loss_video = sum(F.cross_entropy(torch.stack([o[0], r[0]])[None], zero)
                                 for o, r in zip(own, oth)) / len(spans)
                loss = (loss_text + args.video_w * loss_video) * len(spans) / args.batch
                scaler.scale(loss).backward()
                run["text"] += loss_text.item() * len(spans)
                run["video"] += loss_video.item() * len(spans)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0:
                n = 50 * args.batch
                print(f"ep {epoch} step {step}/{total} text {run['text'] / n:.3f} "
                      f"video {run['video'] / n:.3f} {time.time() - t0:.0f}s", flush=True)
                run = defaultdict(float)
        res = pairwise_valid(vlm, valid, feats, args.valid_n)
        print(f"epoch {epoch} valid {json.dumps(res)}", flush=True)
        log.write(json.dumps({"epoch": epoch, "valid": res, "seconds": round(time.time() - t0)}) + "\n")
        log.flush()
        vlm.model.save_pretrained(args.out / f"epoch{epoch}")


# ---------------------------------------------------------------- TemporalBench (eval only)

CELLS = ["single", "ab_ab_c", "ab_ab_ac", "share4"]


@torch.no_grad()
def cmd_eval_tb(args):
    vlm = VLM(args.model, args.device, adapter=args.adapter)
    train = torch.load(args.ref_cache)
    refs = [train[e] for e in random.Random(0).sample(sorted(train), args.refs)]  # fixed Train videos
    del train
    feats = torch.load(args.cache)
    rows = [r for c in CELLS for r in json.loads((args.sets / f"{c}.json").read_text())
            if r["episode"] in feats][:args.limit]
    texts = defaultdict(set)
    for r in rows:
        texts[r["episode"]].update(c["text"] for c in r["choices"])
    by_ds = defaultdict(list)
    for ep, ds in sorted({(r["episode"], r["dataset"]) for r in rows}):
        by_ds[ds].append(ep)
    other = {ep: eps[(i + 1) % len(eps)] for eps in by_ds.values() for i, ep in enumerate(eps)}
    out, t0 = {}, time.time()
    for i, (ep, ts) in enumerate(sorted(texts.items())):
        ts = sorted(ts)
        s = vlm.score_videos([(feats[ep], ts), (feats[other[ep]], ts)] + [(f, ts) for f in refs])
        own, ctrl, ref = s[0].tolist(), s[1].tolist(), torch.stack(s[2:]).mean(0).tolist()
        out[ep] = {t: [a, b, c] for t, a, b, c in zip(ts, own, ctrl, ref)}
        if i % 300 == 0:
            print(f"{i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"adapter": str(args.adapter), "scores": out}) + "\n")
    print(f"{len(out)} episodes -> {args.out} ({time.time() - t0:.0f}s)")


def zscore(x):
    m = sum(x) / len(x)
    sd = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5
    return [(v - m) / (sd + 1e-6) for v in x]


def cmd_report_tb(args):
    sys.path.insert(0, str(ROOT / "diff_probe"))
    from centrality import score_byte, score_span
    scores = json.loads(args.scores.read_text())["scores"]
    rows = [r for c in CELLS for r in json.loads((args.sets / f"{c}.json").read_text())]
    rows = [r for r in rows if all(c["text"] in scores.get(r["episode"], {}) for c in r["choices"])]
    w = json.loads((ROOT / "validator/calibrated/weights_full.json").read_text())["weights"]
    lams = [0.3, 1.0]
    acc = defaultdict(lambda: defaultdict(list))
    pair = defaultdict(lambda: defaultdict(list))
    for r in rows:
        half = "A" if hash_half(r["episode"]) == 0 else "B"
        texts = [c["text"] for c in r["choices"]]  # index 0 is GT
        perm = list(range(5))
        random.Random(f"0:{r['episode']}:{r['cell']}").shuffle(perm)
        span = [a + 1e-3 * b for a, b in zip(score_span(texts), score_byte(texts))]
        sc = [scores[r["episode"]][t] for t in texts]
        for video, j in (("own", 0), ("ctrl", 1)):
            v = [x[j] for x in sc]
            cfgs = {"span": span, "vlm": v}
            cfgs.update({f"span+{lam}vlm": [s + lam * z for s, z in zip(span, zscore(v))] for lam in lams})
            if len(sc[0]) > 2:  # debiased by the fixed Train reference videos
                db = [x[j] - x[2] for x in sc]
                cfgs["db"] = db
                cfgs.update({f"span+{lam}db": [s + lam * z for s, z in zip(span, zscore(db))] for lam in lams})
            for name, s in cfgs.items():
                if video == "ctrl" and name == "span":
                    continue
                rank = sorted(perm, key=lambda i: -s[i]).index(0)  # ties: shuffled order, as the server
                for h in (half, "all"):
                    acc[(r["cell"], h)][f"{video}:{name}"].append(RANK_SCORE[rank])
            if r["cell"] == "single":
                for name in ("vlm", "db"):
                    if name not in cfgs:
                        continue
                    x = cfgs[name]
                    for c, sv in zip(r["choices"][1:], x[1:]):
                        hit = 1.0 if x[0] > sv else 0.5 if x[0] == sv else 0.0
                        pair[c["kind"]][f"{video}:{name}"].append(hit)
                        pair["ALL"][f"{video}:{name}"].append(hit)
    mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731
    print("\n## Pairwise accuracy, single cell (GT vs one negative)\n")
    cols = [c for c in ("own:vlm", "ctrl:vlm", "own:db", "ctrl:db") if c in pair["ALL"]]
    print("| edit category | pairs | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
    for k in sorted(pair, key=lambda k: (k == "ALL", k)):
        print(f"| {k} | {len(pair[k]['own:vlm'])} | " + " | ".join(f"{mean(pair[k][c]):.3f}" for c in cols) + " |")
    names = ["own:span", "own:vlm"] + [f"own:span+{lam}vlm" for lam in lams] + \
        ["ctrl:vlm"] + [f"ctrl:span+{lam}vlm" for lam in lams]
    if "own:db" in pair["ALL"]:
        names += ["own:db"] + [f"own:span+{lam}db" for lam in lams] + [f"ctrl:span+{lam}db" for lam in lams]
    for h in ("A", "B", "all"):
        print(f"\n## 5-choice, half {h}\n")
        print("| cell | n | " + " | ".join(names) + " |\n|---" * 1 + "|---" * (len(names) + 1) + "|")
        mix = defaultdict(float)
        for c in CELLS:
            a = acc[(c, h)]
            print(f"| {c} | {len(a['own:span'])} | " + " | ".join(f"{mean(a[n]):.3f}" for n in names) + " |")
            for n in names:
                mix[n] += w.get(c, 0) * mean(a[n])
        print("| calibrated_mix | - | " + " | ".join(f"{mix[n]:.3f}" for n in names) + " |")


def hash_half(ep):
    return sum(map(ord, ep)) % 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, default=ROOT / "diff_probe/models/smolvlm")
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--limit", type=int)
    p = sub.add_parser("cache", parents=[common])
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("train", parents=[common])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--hf", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=ROOT / "baseline_v2/validation/splits.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--batch", type=int, default=8, help="episodes per optimizer step")
    p.add_argument("--micro", type=int, default=8, help="episodes per forward (lower on a T4)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--video-w", type=float, default=1.0)
    p.add_argument("--loss", choices=["db", "raw"], default="db")
    p.add_argument("--valid-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("eval_tb", parents=[common])
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--ref-cache", type=Path, default=ROOT / "diff_probe/runs/vlm_cache/train.pt")
    p.add_argument("--refs", type=int, default=3)
    p.add_argument("--sets", type=Path, default=TB / "sets")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("report_tb")
    p.add_argument("scores", type=Path)
    p.add_argument("--sets", type=Path, default=TB / "sets")
    args = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "eval_tb": cmd_eval_tb, "report_tb": cmd_report_tb}[args.cmd](args)


if __name__ == "__main__":
    main()
