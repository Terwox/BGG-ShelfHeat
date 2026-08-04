"""Publish ShelfHeat to a Hugging Face Gradio Space.

Requires:
  HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or local .secrets with HF_TOKEN=...
  BGG_API_TOKEN or local .secrets with BGG_API_TOKEN=...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
ALLOW_PATTERNS = [
    "README.md",
    "app.py",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "shelfheat/**",
    "docs/huggingface-space.md",
    "docs/public-release-plan.md",
    ".secrets.example",
]
IGNORE_PATTERNS = [
    ".git/**",
    ".omx/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    ".pytest_cache/**",
    ".secrets",
    ".env",
    ".env.*",
    "output/**",
    "output_eval/**",
    "*.html",
    "results*.json",
    "test-collection.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_id",
        nargs="?",
        help="Hugging Face Space id, e.g. username/shelfheat. Defaults to token-user/shelfheat.",
    )
    parser.add_argument("--space-name", default="shelfheat", help="Space name when repo_id is omitted")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    if args.dry_run:
        repo_id = args.repo_id or f"<HF_USERNAME>/{args.space_name}"
        _print_plan(repo_id, private=args.private)
        return 0

    hf_token = _read_hf_token()
    bgg_token = _read_bgg_token()

    api = HfApi(token=hf_token)
    repo_id = args.repo_id or _default_repo_id(api, hf_token, args.space_name)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    api.add_space_secret(repo_id, "BGG_API_TOKEN", bgg_token)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=ROOT,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Release ShelfHeat public beta",
        commit_description=(
            "Upload Gradio app, server-side BGG API support, docs, and release metadata."
        ),
    )
    print(f"Published ShelfHeat Space: https://huggingface.co/spaces/{repo_id}")
    print(f"Commit: {commit.oid}")
    return 0


def _read_hf_token() -> str:
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or _read_secret_value("HF_TOKEN")
        or _read_secret_value("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or add HF_TOKEN to local .secrets "
            "before publishing the Space."
        )
    return token


def _read_bgg_token() -> str:
    token = os.environ.get("BGG_API_TOKEN", "").strip()
    if token:
        return token

    token = _read_secret_value("BGG_API_TOKEN")
    if token:
        return token

    raise SystemExit("Set BGG_API_TOKEN or add it to local .secrets before publishing.")


def _read_secret_value(name: str) -> str:
    secrets = ROOT / ".secrets"
    if secrets.exists():
        for line in secrets.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() == name and value.strip():
                return value.strip()
    return ""


def _default_repo_id(api: HfApi, token: str, space_name: str) -> str:
    user = api.whoami(token=token, cache=False).get("name")
    if not user:
        raise SystemExit("Could not infer Hugging Face username from the HF token.")
    return f"{user}/{space_name}"


def _print_plan(repo_id: str, *, private: bool) -> None:
    print(f"Would publish Space: {repo_id}")
    print(f"Private: {private}")
    print("Allowed upload patterns:")
    for pattern in ALLOW_PATTERNS:
        print(f"  + {pattern}")
    print("Ignored upload patterns:")
    for pattern in IGNORE_PATTERNS:
        print(f"  - {pattern}")


if __name__ == "__main__":
    raise SystemExit(main())
