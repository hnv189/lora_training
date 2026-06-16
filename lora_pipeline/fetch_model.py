"""Model fetching — download base models from the Hugging Face Hub.

Import-safe: huggingface_hub is only imported inside functions, so the web
app and the rest of the pipeline still run without it installed.

Run with:
    python -m lora_pipeline.fetch_model Qwen/Qwen3-8B
    python -m lora_pipeline.fetch_model Qwen/Qwen3-8B --output ./models/Qwen3-8B --revision main
"""

from __future__ import annotations

import argparse
import os
import threading
from datetime import datetime, timezone
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Only formats the dashboard knows how to load for LoRA training.
SUPPORTED_EXTENSIONS = (".safetensors", ".gguf")

_job: dict[str, Any] = {"status": "idle"}
_lock = threading.Lock()


def deps_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except Exception:
        return False


def local_path_for(repo_id: str) -> str:
    return os.path.join(MODELS_DIR, repo_id.replace("/", "__"))


def search_models(query: str, limit: int = 15) -> list[dict[str, Any]]:
    """Search the Hub, keeping only repos that ship a supported weight format."""
    from huggingface_hub import HfApi

    api = HfApi()
    results: list[dict[str, Any]] = []
    for m in api.list_models(search=query, limit=limit * 3, sort="downloads"):
        try:
            files = api.list_repo_files(m.id)
        except Exception:
            continue
        matched = [f for f in files if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        if not matched:
            continue
        results.append(
            {
                "repo_id": m.id,
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "extensions": sorted({os.path.splitext(f)[1].lower() for f in matched}),
                "local": os.path.isdir(local_path_for(m.id)),
            }
        )
        if len(results) >= limit:
            break
    return results


def download_model(
    repo_id: str,
    local_dir: str | None = None,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Download a full model snapshot to ``local_dir``. Returns the local path."""
    from huggingface_hub import snapshot_download

    local_dir = local_dir or local_path_for(repo_id)
    os.makedirs(local_dir, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=token or os.environ.get("HF_TOKEN"),
    )


def list_local_models() -> list[dict[str, Any]]:
    """List models already downloaded into MODELS_DIR."""
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path):
            out.append({"repo_id": name.replace("__", "/"), "path": path})
    return out


# ---------------------------------------------------------------------------
# Background job — lets the dashboard kick off a download without blocking.
# ---------------------------------------------------------------------------


def status() -> dict[str, Any]:
    with _lock:
        return dict(_job)


def start(
    repo_id: str,
    local_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    with _lock:
        if _job.get("status") == "running":
            raise RuntimeError("A model download is already running.")
        if not deps_available():
            raise RuntimeError(
                "huggingface_hub is not installed. Run: pip install -r requirements.txt"
            )
        job = {
            "status": "running",
            "repo_id": repo_id,
            "local_dir": local_dir or local_path_for(repo_id),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ended_at": None,
            "error": None,
        }
        _job.clear()
        _job.update(job)

    def _run():
        try:
            path = download_model(repo_id, local_dir, revision, token=token)
            with _lock:
                _job["status"] = "completed"
                _job["local_dir"] = path
        except Exception as exc:
            with _lock:
                _job["status"] = "failed"
                _job["error"] = str(exc)
        finally:
            with _lock:
                _job["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    threading.Thread(target=_run, daemon=True).start()
    return status()


def main():
    ap = argparse.ArgumentParser(
        description="Download a model snapshot from the Hugging Face Hub"
    )
    ap.add_argument("repo_id", help="e.g. Qwen/Qwen3-8B")
    ap.add_argument(
        "--output", default=None, help="local directory (default: ./models/<repo_id>)"
    )
    ap.add_argument("--revision", default=None, help="branch, tag, or commit hash")
    ap.add_argument(
        "--token", default=None, help="HF token for gated/private repos (or set HF_TOKEN env var)"
    )
    args = ap.parse_args()

    path = download_model(args.repo_id, args.output, args.revision, token=args.token)
    print(f"Downloaded {args.repo_id} to {path}")


if __name__ == "__main__":
    main()
