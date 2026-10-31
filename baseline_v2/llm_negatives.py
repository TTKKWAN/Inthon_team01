"""Hard negatives written by a local LLM (rules 3-4: generative augmentation, T4-reproducible).

Rule-based edits leave traces (agreement slips, odd word pairs, templates) that a text-only
model can spot; the leaderboard showed those traces do not exist in the test negatives.
Following SugarCrepe / VideoCon, a small instruction model rewrites each GT so it is
wrong in exactly one way while staying fluent. The code only holds the instructions; every
sentence comes from the model.

  model   Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0), float16 (~8 GB VRAM: fits a T4 16 GB;
          T4 has no bfloat16), greedy decoding -> deterministic for a given revision
  output  JSON like negatives.json: {episodes: {ep: [{"pos", "negs": [[type, text], ...]}]}}

  python llm_negatives.py --out cache/llm_negatives.json                 # every episode, full GT
  python llm_negatives.py --out cache/llm_negatives.json --first-sentence --limit 20
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from negatives import segment, sentences

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"  # pinned for reproduction

SYSTEM = ("You edit descriptions of short videos to make subtly wrong answer options for a "
          "video understanding test.")
TASK = ("Original description:\n{text}\n\n"
        "Rewrite it so that it becomes wrong about the video in exactly one way: {how}. "
        "Keep every other detail, the sentence structure, tense and style the same, and keep "
        "it fluent and natural so the change is not obvious from the wording alone. "
        "Output only the rewritten description.")
EDITS = {
    "llm_action": "replace one action with a different action that could plausibly be done "
                  "with the same object (for example its opposite)",
    "llm_order": "swap the order in which two of the described events happen",
    "llm_object": "replace one object with a different everyday object",
    "llm_attribute": "change one color, size, count or position (such as left or right) of "
                     "an object",
    "llm_add": "add one extra event that does not happen in the video, using an object that "
               "is already mentioned",
}


def load_gt(data_dir):
    """{episode: GT description} straight from Train/episode_*/description.txt."""
    folders = sorted((p for p in Path(data_dir).glob("episode_*") if p.is_dir()),
                     key=lambda p: int(p.name.split("_")[1]))
    return {f.name: (f / "description.txt").read_text(encoding="utf-8").strip()
            for f in folders}


def build_jobs(gts, episodes, first_sentence):
    """One job per (episode, positive, edit type). Positive 0 = the full GT; positive 1 =
    its first sentence (only with --first-sentence and when the GT has >= 2 sentences)."""
    jobs = []
    for ep in episodes:
        positives = [gts[ep]]
        if first_sentence and len(sentences(gts[ep])) >= 2:
            positives.append(sentences(gts[ep])[0])
        for gi, pos in enumerate(positives):
            n_events = len(segment(pos)[0])
            for kind, how in EDITS.items():
                if kind == "llm_order" and n_events < 2:
                    continue
                jobs.append((ep, gi, kind, pos, how))
    return jobs


def clean(output, original):
    text = output.strip().strip('"').strip()
    text = re.sub(r"^(?:rewritten description|answer)\s*:\s*", "", text, flags=re.I).strip()
    text = text.split("\n\n")[0].strip()
    if not text or text.lower() == original.lower():
        return None
    # Far longer or shorter than the original = the model rewrote everything.
    ratio = len(text) / max(len(original), 1)
    if not 0.6 < ratio < 1.8:
        return None
    return text


def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                 dtype=torch.float16).cuda().eval()
    return tok, model


def generate(tok, model, user_prompts, sources, batch_size=48):
    """Greedy completions for each user prompt (same system prompt), in input order.
    `sources` = the text each prompt rewrites, used to size max_new_tokens."""
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in user_prompts]
    outputs = [None] * len(prompts)
    t0 = time.time()
    # Longest prompts first: less padding per batch.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for s in range(0, len(order), batch_size):
        batch = order[s:s + batch_size]
        enc = tok([prompts[i] for i in batch], return_tensors="pt", padding=True).to("cuda")
        longest = max(len(tok(sources[i])["input_ids"]) for i in batch)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=int(longest * 1.6) + 24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(batch, texts):
            outputs[i] = t
        if (s // batch_size) % 10 == 0:
            done = min(s + batch_size, len(order))
            print(f"{done}/{len(prompts)} generated, {time.time() - t0:.0f}s", flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/qwen3-4b-instruct-2507")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "../data/Train")
    parser.add_argument("--episodes", default="all", help="all, or splits.json:train|valid")
    parser.add_argument("--first-sentence", action="store_true",
                        help="also make negatives for the first sentence as a short positive")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit", type=int, help="first N episodes (smoke test)")
    args = parser.parse_args()
    gts = load_gt(args.data_dir)
    if args.episodes == "all":
        episodes = list(gts)
    else:
        split_file, part = args.episodes.split(":")
        episodes = json.loads((ROOT / split_file).read_text(encoding="utf-8"))[part]
    if args.limit:
        episodes = episodes[:args.limit]
    jobs = build_jobs(gts, episodes, args.first_sentence)
    tok, model = load_model(args.model_dir)
    t0 = time.time()
    outputs = generate(tok, model, [TASK.format(text=text, how=how)
                                    for _, _, _, text, how in jobs],
                       [text for _, _, _, text, _ in jobs], args.batch_size)
    out, kept = {}, 0
    for (ep, gi, kind, original, _), raw in zip(jobs, outputs):
        groups = out.setdefault(ep, [])
        if len(groups) <= gi:
            groups.append({"pos": original, "negs": []})
        neg = clean(raw, original)
        if neg:
            groups[gi]["negs"].append([kind, neg])
            kept += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION,
                                    "dtype": "float16", "decoding": "greedy",
                                    "instructions": EDITS, "episodes": out},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: {kept}/{len(jobs)} kept in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
