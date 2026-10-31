"""Offline local validation, optionally using the real server archive contract and runner.

Only reads original Train images/descriptions. Test data is never used.
This is an interface check, not a leaderboard accuracy measurement.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import ROOT, choose_device
from dataset import load_episodes, load_images


def check_ranking(ranking, choices):
    ids = [choice["id"] for choice in choices]
    if (not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking)
            or len(ranking) != len(ids) or set(ranking) != set(ids)):
        raise ValueError(f"Ranking must contain every candidate ID once: {ranking}")


def check_direct(artifact, data_dir, device):
    sys.path.insert(0, str(artifact))
    spec = importlib.util.spec_from_file_location("participant_submission", artifact / "submission.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    predictor = module.Submission(artifact_dir=str(artifact), device=device)
    episodes = load_episodes(data_dir)
    if len(episodes) < 2:
        raise ValueError("Validation requires at least two original Train episodes.")
    episode = episodes[0]
    choices = [{"id": f"choice-{i}", "text": item.description}
               for i, item in enumerate(episodes[:5])]
    images = load_images(episode)
    try:
        scores = predictor.score(images, choices)
        if len(scores) != len(choices) or any(not 0 <= score <= 1 for score in scores):
            raise ValueError(f"Expected one probability in [0, 1] per choice: {scores}")
        ranking = predictor.predict(images, choices)
        check_ranking(ranking, choices)
        # Also exercise variable candidate counts, supplied IDs and repeated calls.
        reordered = list(reversed(choices[:2]))
        check_ranking(predictor.predict(images, reordered), reordered)
        if ranking != predictor.predict(images, choices):
            raise ValueError("Repeated calls changed the prediction.")
    finally:
        for image in images:
            image.close()
    print(json.dumps({"interface": "passed", "episode": episode.episode_id,
                      "scores": scores, "ranking": ranking}, indent=2), flush=True)


def check_server(artifact, data_dir, server_dir, device, work_dir):
    episodes = load_episodes(data_dir)
    if len(episodes) < 2:
        raise ValueError("Validation requires at least two original Train episodes.")
    inputs = work_dir / "inputs"
    inputs.mkdir()
    requests = []
    for episode in episodes[:2]:
        frames = []
        for path in episode.frames:
            relative = Path(episode.episode_id) / path.name
            target = inputs / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            frames.append({"path": relative.as_posix()})
        requests.append({
            "episode_id": episode.episode_id, "images": frames,
            "choices": [{"id": f"choice-{i}", "text": item.description}
                        for i, item in enumerate(episodes[:5])],
        })
    (inputs / "inputs.json").write_text(json.dumps({
        "version": "starterkit-train-interface-check", "episodes": requests,
    }), encoding="utf-8")
    plan = work_dir / "plan.json"
    plan.write_text(json.dumps([[item["episode_id"]] for item in requests]), encoding="utf-8")
    output = work_dir / "runner-output"
    # -I, a new CWD and an empty HF cache detect accidental dependencies on the kit or user cache.
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HOME": str(work_dir / "empty-hf-cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([
        sys.executable, "-I", str(server_dir / "runner/execute.py"),
        "--artifact", str(artifact), "--inputs", str(inputs),
        "--plan", str(plan), "--output", str(output), "--device", device,
    ], cwd=work_dir, env=environment, check=False)
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "batches.jsonl").read_text().splitlines()]
    if completed.returncode or summary.get("fatal_error") or len(rows) != len(requests):
        raise RuntimeError(f"Server runner failed: {summary}")
    for row, request in zip(rows, requests):
        prediction = row["predictions"][0]
        if "ranking" not in prediction:
            raise RuntimeError(f"Server prediction failed: {prediction}")
        check_ranking(prediction["ranking"], request["choices"])
    print(json.dumps({"server_runner": "passed", "problems": len(rows),
                      "api_version": summary["api_version"],
                      "prediction_interface": summary["prediction_interface"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "sample_submission")
    parser.add_argument("--zip", type=Path, help="Validate an extracted copy of this ZIP.")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data/Train")
    parser.add_argument("--server-dir", type=Path, help="Use server/contracts and the actual server runner.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.zip and not args.server_dir:
        parser.error("--zip requires --server-dir for safe extraction with the server contract.")
    with tempfile.TemporaryDirectory(prefix="starterkit-check-") as temp:
        work = Path(temp)
        artifact = args.submission_dir.resolve()
        if args.server_dir:
            server_dir = args.server_dir.resolve()
            if args.zip:
                sys.path.insert(0, str(server_dir))
                from contracts.archive import extract_zip, validate_artifact
                print(f"Server ZIP contract: {validate_artifact(args.zip)}", flush=True)
                artifact = work / "artifact"
                extract_zip(args.zip, artifact)
            check_server(artifact, args.data_dir.resolve(), server_dir, device, work)
        else:
            check_direct(artifact, args.data_dir.resolve(), device)


if __name__ == "__main__":
    main()
