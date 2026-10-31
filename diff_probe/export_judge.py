"""Build the span + video-judge submission folder and ZIP (diff_probe/video_judge/submission.py).

Ships DINOv2 and SigLIP (fp16, no MPNet), the pair head checkpoint, and the 64 reference
videos' cached frame features used for the text priors (Train episodes listed in the
rerank_eval.py output, fixed before any test input is seen).

    python diff_probe/export_judge.py --head diff_probe/runs/pair_head/epoch09.pt \
        --refs diff_probe/runs/rerank_eval.json --l1 0.3 --l2 0.3 --out diff_probe/submissions/video_judge
    python diff_probe/export_judge.py --head-type patch --head diff_probe/runs/patch_head_x/epoch11.pt \
        --refs diff_probe/runs/rerank_eval.json --out diff_probe/submissions/video_judge_patch
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline_v2"))
from model_v2 import FrozenEncoders  # noqa: E402

SPAN_FILES = ROOT / "diff_probe/submissions/span_centrality"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True, help="rerank_eval.py --out json (ref_eps)")
    ap.add_argument("--head-type", choices=["pair", "patch"], default="pair")
    ap.add_argument("--frames", type=Path, default=ROOT / "baseline_v2/cache/frames.pt")
    ap.add_argument("--patches", type=Path, default=ROOT / "baseline_v2/cache/patches4.pt")
    ap.add_argument("--span-dir", type=Path, default=SPAN_FILES)
    ap.add_argument("--l1", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty directory: {out}")

    cfg = json.loads((ROOT / "baseline_v2/config.json").read_text())
    base = ROOT / "baseline_v2"
    enc = FrozenEncoders(base / cfg["image_model_dir"], base / cfg["text_model_dir"],
                         base / cfg["siglip_model_dir"], "cpu")
    models = out / "models"
    for name, model in [("dinov2", enc.image_model), ("siglip", enc.siglip)]:
        state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
                 for k, v in model.state_dict().items()}
        model.save_pretrained(models / name, state_dict=state, safe_serialization=True)
    enc.dino_processor.save_pretrained(models / "dinov2")
    enc.siglip_processor.save_pretrained(models / "siglip")
    enc.siglip_tokenizer.save_pretrained(models / "siglip")

    (out / "weights").mkdir(parents=True)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    torch.save(ckpt["state_dict"], out / "weights/head.pt")
    ref_eps = json.loads(args.refs.read_text())["ref_eps"]
    cache = torch.load(args.frames, map_location="cpu", weights_only=False)
    index, off = {e: i for i, e in enumerate(cache["episode_ids"])}, cache["frame_offsets"]
    refs = [cache["frames"][off[index[e]]:off[index[e] + 1]].clone() for e in ref_eps]
    saved = {"episode_ids": ref_eps, "frames": refs}
    if args.head_type == "patch":
        pc = torch.load(args.patches, map_location="cpu", weights_only=False)
        assert pc["episode_ids"] == cache["episode_ids"] and torch.equal(pc["frame_offsets"], off)
        saved["patches"] = [pc["patches"][off[index[e]]:off[index[e] + 1]].clone() for e in ref_eps]
    torch.save(saved, out / "weights/refs.pt")

    shutil.copyfile(ROOT / "diff_probe/video_judge/submission.py", out / "submission.py")
    shutil.copyfile(ROOT / "diff_probe/video_judge/judge.py", out / "judge.py")
    shutil.copyfile(ROOT / "baseline_v2/model_v2.py", out / "model_v2.py")
    for name in ("centrality.py", "hyuwooma_bytes.py"):
        shutil.copyfile(args.span_dir / name, out / name)
    (out / "submission.json").write_text(json.dumps({"api_version": 2}), encoding="utf-8")
    (out / "inference_config.json").write_text(json.dumps(
        {"dinov2_dir": "models/dinov2", "siglip_dir": "models/siglip", "tmax": ckpt["tmax"],
         "head_type": args.head_type,
         "l1": args.l1, "l2": args.l2, "cpu_threads": 4}, indent=2), encoding="utf-8")
    (out / "training_info.json").write_text(json.dumps(
        {"head": str(args.head), "head_epoch": ckpt["epoch"], "ref_eps": ref_eps,
         "negatives": "HF SANGchu-03/INThon_Augmented qwen/qwen_negatives_from_gt.json "
                      "(Qwen3-4B-Instruct-2507 fp16 greedy), train split of baseline_v2/validation/splits.json",
         "train_cmd": {"pair": "python diff_probe/pair_head.py --hf qwen_negatives_from_gt.json --out diff_probe/runs/pair_head",
                       "patch": "python diff_probe/cache_patches.py && python diff_probe/patch_head.py --hf "
                                "qwen_negatives_from_gt.json --exclude ref_eps.json --out diff_probe/runs/patch_head_x"}[args.head_type]},
        indent=2), encoding="utf-8")

    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, p.relative_to(out).as_posix())
    print(f"Exported {out} and {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
