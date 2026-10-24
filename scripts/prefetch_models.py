"""Download every model before the notebooks run, with retries, so a stalled download shows up
in logs\\download.log instead of hanging a notebook. Safe to re-run: cached files are skipped."""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # plain HTTP: errors out instead of hanging
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from pi_hardening import config  # noqa: E402

PATTERNS = ["*.json", "*.model", "*.txt", "*.py", "*.safetensors", "*.jinja", "*.tiktoken", "merges.txt"]

ok = True
for key, repo in config.MODELS.items():
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            path = snapshot_download(repo, allow_patterns=PATTERNS)
            print(f"OK   {repo} ({time.time() - t0:.0f}s) -> {path}", flush=True)
            break
        except Exception as e:  # network errors: wait and retry
            print(f"FAIL {repo} attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            time.sleep(20)
    else:
        ok = False
sys.exit(0 if ok else 1)
