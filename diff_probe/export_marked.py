"""Build a submission folder + ZIP from a train_marked checkpoint, then check it.

The check imports the exported folder's own Submission, calls predict(images, choices)
the way the server does (frames as RGB PIL images in numeric order), scores half B of the
validation sets with the competition metric, and times each episode.

    python diff_probe/export_marked.py --checkpoint diff_probe/runs/marked_v1/epoch14.pt \
        --marks ref --output-dir diff_probe/submissions/marked_v1_ep14
"""

import argparse
import json
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "baseline_v2"))
from model_v2 import FrozenEncoders  # noqa: E402

RANK_SCORE = [1.0, 0.5, 0.25, 0.125, 0.0]
CODE = {"submission.py": HERE / "submission_marked.py", "marked_model.py": HERE / "marked_model.py",
        "marks.py": HERE / "marks.py", "spots.py": HERE / "spots.py",
        "centrality.py": HERE / "centrality.py", "model_v2.py": ROOT / "baseline_v2/model_v2.py"}
CHECK_SETS = ["short/H1_disjoint", "short/H2_samespot", "short/H6_fact_a_b_ab_c",
              "short/H7_fact_a_b_ab_ac", "full/H5_mix"]


def natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def export(args):
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty directory: {out}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enc = FrozenEncoders(ROOT / "starterkit/models/dinov2", ROOT / "starterkit/models/mpnet",
                         ROOT / "baseline_v2/models/siglip", "cpu")
    out.mkdir(parents=True, exist_ok=True)
    enc.export(out / "models")
    (out / "weights").mkdir()
    torch.save(ckpt["model"], out / "weights/verifier.pt")
    for name, src in CODE.items():
        shutil.copyfile(src, out / name)
    run_args = ckpt.get("args", {})
    (out / "submission.json").write_text(json.dumps({"api_version": 2}), encoding="utf-8")
    (out / "inference_config.json").write_text(json.dumps({
        "model_config": {"max_frames": 32}, "max_frames": run_args.get("max_frames", 16),
        "marks": args.marks, "max_text_tokens": 256, "cpu_threads": 4}, indent=2), encoding="utf-8")
    (out / "training_info.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"), "step": ckpt.get("step"),
        "train_args": run_args}, indent=2, default=str), encoding="utf-8")
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(out).as_posix())
    print(f"Exported {out} and {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return zip_path


def check(args):
    folder = Path(args.output_dir)
    for name in ["submission", "marked_model", "marks", "spots", "centrality", "model_v2"]:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(folder))
    from submission import Submission
    sub = Submission(str(folder), args.device)
    valsets = json.loads((HERE / "valsets.json").read_text())["sets"]
    eps = sorted({r["episode"] for r in valsets["short/H1_disjoint"]}, key=lambda e: int(e.split("_")[1]))
    half_b = set(eps[1::2])
    data = ROOT / "data/Train"
    rng = random.Random(0)
    report, times = {}, []
    for name in CHECK_SETS:
        rows = [r for r in valsets[name] if r["episode"] in half_b][:args.check_limit]
        total = 0.0
        for r in rows:
            folder_imgs = data / r["episode"] / "images"
            paths = sorted((p for p in folder_imgs.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                           key=natural_key)
            images = [Image.open(p).convert("RGB") for p in paths]
            choices = [{"id": "ABCDE"[i], "text": c["text"], "pos": c["type"] == "pos"}
                       for i, c in enumerate(r["choices"])]
            rng.shuffle(choices)
            truth = next(c["id"] for c in choices if c["pos"])
            t0 = time.time()
            order = sub.predict(images, [{"id": c["id"], "text": c["text"]} for c in choices])
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            total += RANK_SCORE[order.index(truth)]
        report[name] = round(total / len(rows), 4)
        print(name, report[name], f"n={len(rows)}", flush=True)
    per_ep = sum(times[5:]) / max(len(times) - 5, 1)
    print(json.dumps({"scores_half_B": report, "sec_per_episode": round(per_ep, 4),
                      "est_minutes_1910_this_gpu": round(per_ep * 1910 / 60, 2)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--marks", choices=["slots", "ref"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-limit", type=int, default=150)
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    if not args.skip_export:
        export(args)
    check(args)


if __name__ == "__main__":
    main()
