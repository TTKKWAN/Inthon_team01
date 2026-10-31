"""List every third-party package a submission ZIP actually loads (init + one predict),
and flag any that is not in the scoring environment.

  python check_deps.py outputs/minimal_submission.zip outputs/v2_debias_submission.zip

Runs each ZIP in a fresh interpreter so imports from earlier checks do not leak in.
"""

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Scoring environment (rules, 제1조): distribution name -> import names.
ALLOWED = {
    "torch": ["torch"], "torchvision": ["torchvision"], "transformers": ["transformers"],
    "numpy": ["numpy"], "Pillow": ["PIL"], "safetensors": ["safetensors"],
    "sentence-transformers": ["sentence_transformers"], "einops": ["einops"],
    "timm": ["timm"], "scipy": ["scipy"], "scikit-learn": ["sklearn"],
    "opencv-python-headless": ["cv2"], "peft": ["peft"], "accelerate": ["accelerate"],
    "bitsandbytes": ["bitsandbytes"],
}
# Installed alongside the allowed packages as their own dependencies, so present on the server.
DEPENDENCIES_OF_ALLOWED = {
    "huggingface_hub", "tokenizers", "regex", "yaml", "filelock", "fsspec", "packaging",
    "requests", "urllib3", "certifi", "charset_normalizer", "idna", "tqdm", "jinja2",
    "markupsafe", "sympy", "mpmath", "networkx", "typing_extensions", "psutil", "triton",
    "nvidia", "hf_xet", "colorama", "setuptools", "pkg_resources", "_distutils_hack",
}

CHILD = r"""
import sys, json
before = set(sys.modules)
sys.path.insert(0, sys.argv[1])
from PIL import Image
from submission import Submission
sub = Submission(sys.argv[1], "cuda")
imgs = [Image.new("RGB", (640, 360), (i * 20, 80, 120)) for i in range(8)]
sub.predict(imgs, [{"id": "a", "text": "A person lifts a cup and then puts it down."},
                   {"id": "b", "text": "They are holding a red box."}])
roots = sorted({m.split(".")[0] for m in set(sys.modules) - before})
print(json.dumps(roots))
"""


def main():
    stdlib = set(sys.stdlib_module_names)
    allowed = {name for names in ALLOWED.values() for name in names}
    for path in sys.argv[1:]:
        folder = Path(tempfile.mkdtemp(prefix="deps_"))
        with zipfile.ZipFile(path) as z:
            z.extractall(folder)
        own = {p.stem for p in folder.glob("*.py")}
        out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(folder)],
                             capture_output=True, text=True, cwd=folder)
        if out.returncode:
            print(f"{path}: FAILED\n{out.stderr[-2000:]}")
            continue
        roots = json.loads(out.stdout.strip().splitlines()[-1])
        third = [r for r in roots if r not in stdlib and r not in own and not r.startswith("_")]
        bad = [r for r in third if r not in allowed and r not in DEPENDENCIES_OF_ALLOWED]
        print(f"{Path(path).name}")
        print(f"  allowed packages used : {sorted(r for r in third if r in allowed)}")
        print(f"  their dependencies    : {sorted(r for r in third if r in DEPENDENCIES_OF_ALLOWED)}")
        print(f"  NOT ALLOWED           : {bad or 'none'}")


if __name__ == "__main__":
    main()
