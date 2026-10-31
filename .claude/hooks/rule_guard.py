#!/usr/bin/env python3
"""PreToolUse guard for the Inthon competition rules (see Rule.md).

Blocks (deny) or asks for confirmation on tool calls that would likely
violate the rules. Reads the hook JSON on stdin and prints a decision JSON.
"""
import json
import os
import re
import sys

ALLOWED_MODELS = {
    "facebook/dinov2-base",
    "sentence-transformers/all-mpnet-base-v2",
    "google/siglip-base-patch16-224",
    "huggingfacetb/smolvlm-256m-instruct",
}
ALLOWED_NAMES = {m.split("/")[-1] for m in ALLOWED_MODELS}

# Generative models are allowed only for data augmentation (Rule 3.4).
GENERATIVE_RE = re.compile(
    r"instruct|chat|-it\b|qwen|llama|gemma|mistral|phi-?\d|gpt|diffusion|stable-?diffusion|sdxl|flux|kandinsky|pixart",
    re.I,
)
# Non-generative pretrained encoders/backbones that are NOT on the allowlist.
ENCODER_RE = re.compile(
    r"clip|bert|roberta|deberta|electra|xlm|\bvit|deit|beit|swin|convnext|resnet|efficientnet|regnet|mobilenet|"
    r"densenet|inception|eva|blip|dino|siglip|mpnet|minilm|e5-|bge-|gte-|instructor|sentence-t5|t5-|bart|"
    r"layoutlm|owl|sam-|detr|yolo|smolvlm|paligemma|florence|idefics|llava",
    re.I,
)
MAX_GEN_PARAMS_B = 20  # > ~20B cannot be reproduced on a 16GB T4 even when quantized

MODEL_CALL_RE = re.compile(
    r"(?:from_pretrained|SentenceTransformer|CrossEncoder|snapshot_download|hf_hub_download|pipeline|"
    r"create_model|open_clip\.\w+|load_model)\s*\(\s*(?:[A-Za-z_]+\s*=\s*)?[rfb]?[\"']([^\"']+)[\"']"
)
MODEL_ASSIGN_RE = re.compile(
    r"\b(?:model_name|model_id|repo_id|pretrained_model_name_or_path|base_model|backbone|encoder_name)\w*\s*[:=]\s*[rf]?[\"']([^\"']+)[\"']",
    re.I,
)
CLI_DOWNLOAD_RE = re.compile(r"\b(?:huggingface-cli|hf)\s+download\s+([^\s]+)(.*)")
PRETRAINED_FLAG_RE = re.compile(
    r"pretrained\s*=\s*True|weights\s*=\s*[\"']?(?:IMAGENET|DEFAULT)|weights\s*=\s*\w+_Weights\.|"
    r"torch\.hub\.load\s*\(",
)

EXTERNAL_API_RE = re.compile(
    r"^\s*(?:import|from)\s+(?:openai|anthropic|google\.generativeai|google\.genai|cohere|groq|together|mistralai|litellm)\b|"
    r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|ChatOpenAI|ChatAnthropic",
    re.M,
)
DISTILL_RE = re.compile(
    r"distil(?!bert|roberta|gpt)|teacher[_ ]?(?:model|logit|prob|pred|output|score|rank|label)|"
    r"soft[_ ]?(?:label|target)|knowledge[_ ]distillation",
    re.I,
)
EXTERNAL_DATA_RE = re.compile(r"load_dataset\s*\(\s*[\"'](?![./~])([^\"']+)[\"']")
TRAIN_STEP_RE = re.compile(r"\.backward\s*\(|optimizer\.step\s*\(|\.step\s*\(\s*\)\s*#?.*optim|requires_grad_\(\s*True")

# Hand-crafted descriptions are banned (organizer notice): many English sentence literals in code.
SENTENCE_LITERAL_RE = re.compile(r"[\"']([A-Z][a-z]*(?: [A-Za-z0-9',-]+){4,}[.!?]?)[\"']")
HANDCRAFTED_THRESHOLD = 5

CODE_EXTS =(".py", ".ipynb", ".sh", ".yaml", ".yml", ".toml", ".cfg")
DATA_EXTS = (".jsonl", ".csv", ".tsv", ".parquet", ".arrow", ".npy", ".npz", ".pkl", ".pt")
DATAISH_PATH_RE = re.compile(r"(?:^|/)(?:data|dataset|datasets|aug\w*|episode\w*|train\w*|label\w*|generated)(?:/|[_.-])", re.I)
INFER_PATH_RE = re.compile(r"infer|predict|submission|submit|inference", re.I)

GRADING_PKGS = {
    "torch", "torchvision", "transformers", "numpy", "pillow", "safetensors", "sentence-transformers",
    "einops", "timm", "scipy", "scikit-learn", "sklearn", "opencv-python-headless", "peft", "accelerate",
    "bitsandbytes", "pip", "setuptools", "wheel",
}


def decide(kind, reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": kind,
            "permissionDecisionReason": f"[Rule.md 가드] {reason}",
        }
    }, ensure_ascii=False))
    sys.exit(0)


def is_local_path(s):
    return s.startswith((".", "/", "~", "$")) or "{" in s or s.count("/") > 1 or re.search(r"checkpoint|ckpt|output|runs?/", s, re.I)


def check_model_id(raw):
    mid = raw.strip()
    if not mid or is_local_path(mid):
        return None
    low = mid.lower()
    if low in ALLOWED_MODELS or low.split("/")[-1] in ALLOWED_NAMES:
        return None
    if GENERATIVE_RE.search(low) and not re.search(r"smolvlm", low):
        sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", low)]
        if sizes and max(sizes) > MAX_GEN_PARAMS_B:
            return ("deny", f"생성형 모델 '{mid}' 는 T4(16GB)에서 재현할 수 없다(제3조 4·5항). {MAX_GEN_PARAMS_B}B 이하 모델을 양자화해 써라.")
        return ("ask", f"생성형 모델 '{mid}' 는 데이터 증강 용도로만 쓸 수 있다(제3조 4항). 제출 모델이나 추론 코드에 쓰는 거라면 금지(제1조 2항). T4 16GB에서 돌아가는지도 확인할 것.")
    if ENCODER_RE.search(low):
        return ("deny", f"허용되지 않은 사전학습 모델 '{mid}'(제1조 2항). 허용 목록: {', '.join(sorted(ALLOWED_MODELS))}")
    return ("ask", f"허용 목록에 없는 모델 ID '{mid}'. 허용된 4개 모델이나 증강용 생성형 모델이 아니면 제1조 2항 위반이다.")


def scan_models(text):
    hits = [m.group(1) for m in MODEL_CALL_RE.finditer(text)] + [m.group(1) for m in MODEL_ASSIGN_RE.finditer(text)]
    asks = []
    for h in hits:
        r = check_model_id(h)
        if r and r[0] == "deny":
            decide(*r)
        if r:
            asks.append(r[1])
    return asks


def check_code(path, text):
    asks = []
    if EXTERNAL_API_RE.search(text):
        decide("deny", "외부 생성형 AI API(OpenAI, Anthropic, Gemini 등) 사용 금지. 추론에 쓰는 것(제2조 4항)과 데이터 생성에 쓰는 것(제3조 6항) 모두 해당.")
    if DISTILL_RE.search(text):
        decide("deny", "모델 증류(teacher 예측, logits, soft label을 학습 목표로 쓰는 것) 금지(제4조).")
    m = EXTERNAL_DATA_RE.search(text)
    if m:
        decide("deny", f"외부 데이터셋 '{m.group(1)}' 사용 금지. 운영진 제공 train data와 허용된 증강 데이터만 쓸 수 있다(제3조 7항).")
    if PRETRAINED_FLAG_RE.search(text) and not re.search(r"pretrained\s*=\s*True[^\n]*#\s*rule-ok", text):
        decide("deny", "timm/torchvision/torch.hub의 사전학습 가중치(pretrained=True, IMAGENET weights 등)는 허용 목록에 없다(제1조 2항). 허용된 4개 모델만 쓸 것.")
    if re.search(r"requests\.(?:post|put)\s*\([^)]*files\s*=", text):
        asks.append("HTTP 파일 업로드 코드. 리더보드 제출용이면 금지다(사용자가 직접 제출).")
    if path and INFER_PATH_RE.search(path) and TRAIN_STEP_RE.search(text):
        decide("deny", "추론 코드에서 파라미터 학습(backward, optimizer.step) 금지(제2조 3항).")
    sentences = [s for s in SENTENCE_LITERAL_RE.findall(text) if len(s.split()) >= 6]
    if len(sentences) >= HANDCRAFTED_THRESHOLD:
        asks.append(f"코드에 영어 문장 리터럴 {len(sentences)}개(예: \"{sentences[0][:60]}\"). hand-crafted description은 금지다(운영진 공지, 제3조 6항). 생성형 모델 프롬프트 템플릿이면 괜찮다.")
    asks += scan_models(text)
    return asks


def check_data_write(path):
    p = path or ""
    if p.lower().endswith(DATA_EXTS) or (p.lower().endswith((".json", ".txt")) and DATAISH_PATH_RE.search(p)):
        decide("deny", f"Claude가 데이터 파일 '{p}' 를 직접 작성하는 것 금지(제3조 6항, 제5조 2항). 증강 코드를 작성하고 그 코드로 데이터를 생성할 것.")


def check_bash(cmd):
    asks = []
    # The user submits to the leaderboard themselves; never upload/submit.
    if re.search(r"\b(?:curl|http|xh)\b[^\n]*(?:\s-F\b|--form\b|\s-T\b|--upload-file\b|--data-binary\s+@|\s-d\s+@|\S+=@)", cmd) or \
       re.search(r"\bkaggle\s+competitions\s+submit\b|\bwget\b[^\n]*--post-file", cmd):
        decide("deny", "제출과 파일 업로드는 사용자가 직접 한다. Claude가 제출하지 말 것(사용자 지시).")
    if re.search(r"\b(?:python3?|bash|sh|\./)\s*\S*submit\S*", cmd, re.I):
        asks.append("'submit'이 들어간 스크립트 실행. 리더보드에 제출하는 동작이면 금지다(사용자가 직접 제출). 로컬에서 예측 파일만 만드는 거라면 괜찮다.")
    if re.search(r"\b(?:cat|echo|printf)\b[^|;&]*(?:>>?|\|\s*tee)\s*\S+\.(?:jsonl|csv|tsv)\b", cmd) or \
       re.search(r"(?:cat|echo|printf)\b[^\n]*>>?\s*\S*(?:data|aug|episode|train|label)\S*\.(?:json|txt)\b", cmd, re.I):
        decide("deny", "Claude가 쉘로 데이터 파일을 직접 작성하는 것 금지(제3조 6항). 증강 스크립트로 생성할 것.")
    if re.search(r"\bkaggle\s+(?:datasets|competitions)\s+download|--repo-type[= ]dataset|load_dataset\s*\(\s*[\"'](?![./~])", cmd):
        decide("deny", "외부 데이터셋 다운로드 금지(제3조 7항).")
    if re.search(r"\b(?:pip3?|uv\s+pip|conda)\s+install\b[^\n]*\b(?:openai|anthropic|google-generativeai|google-genai|cohere|groq|litellm)\b", cmd):
        decide("deny", "외부 생성형 AI API 클라이언트 설치 금지(제2조 4항, 제3조 6항).")
    if re.search(r"\b(?:curl|wget|http)\b[^\n]*(?:api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com)", cmd):
        decide("deny", "외부 생성형 AI API 호출 금지(제2조 4항, 제3조 6항).")
    for m in CLI_DOWNLOAD_RE.finditer(cmd):
        r = check_model_id(m.group(1))
        if r and r[0] == "deny":
            decide(*r)
        if r:
            asks.append(r[1])
    if re.search(r"\bpython3?\s+-c\b|<<", cmd):
        asks += check_code(None, cmd)
    m = re.search(r"\b(?:pip3?|uv\s+pip)\s+install\s+(.+)", cmd)
    if m:
        pkgs = [re.split(r"[=<>\[~!]", t)[0].lower() for t in m.group(1).split() if not t.startswith("-")]
        extra = [p for p in pkgs if p and p not in GRADING_PKGS and not p.startswith((".", "/", "git+"))]
        if extra:
            asks.append(f"채점 환경에 없는 패키지 {extra}. 추론 코드에서 쓰면 제출이 동작하지 않는다(제1조 3항). 증강·분석 용도만 가능.")
    if re.search(r"\bgit\s+clone\b", cmd):
        asks.append("외부 repo clone. 참고만 하고 코드를 그대로 가져오면 표절이다(제5조 3항).")
    return asks


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}
    asks = []

    if tool == "Bash":
        asks += check_bash(ti.get("command", ""))
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = ti.get("file_path") or ti.get("notebook_path") or ""
        project = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ""
        outside = project and os.path.isabs(path) and not os.path.abspath(path).startswith(os.path.abspath(project) + os.sep)
        if outside or "/.claude/" in path or "/.omc/" in path or path.endswith(".md"):
            sys.exit(0)
        check_data_write(path)
        if tool == "Write":
            text = ti.get("content", "")
        elif tool == "Edit":
            text = ti.get("new_string", "")
        elif tool == "MultiEdit":
            text = "\n".join(e.get("new_string", "") for e in ti.get("edits", []))
        else:
            text = ti.get("new_source", "")
        if path.lower().endswith(CODE_EXTS) or tool == "NotebookEdit":
            asks += check_code(path, text)

    if asks:
        decide("ask", " / ".join(dict.fromkeys(asks)))
    sys.exit(0)


if __name__ == "__main__":
    main()
