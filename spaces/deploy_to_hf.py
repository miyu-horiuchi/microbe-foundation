from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


ROOT = Path(__file__).resolve().parent

SPACES = {
    "public": {
        "repo_id": "miyuiu/microbe-foundation",
        "path": ROOT / "public_tool",
    },
    "research": {
        "repo_id": "miyuiu/predictability-gradient",
        "path": ROOT / "research_showcase",
    },
}


def deploy_one(name: str) -> None:
    cfg = SPACES[name]
    repo_id = cfg["repo_id"]
    folder = cfg["path"]
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(folder),
        path_in_repo=".",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
        commit_message=f"Deploy {name} Space",
    )
    print(f"https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the microbe-foundation Spaces to Hugging Face.")
    parser.add_argument("target", choices=["public", "research", "all"], nargs="?", default="all")
    args = parser.parse_args()

    targets = ["public", "research"] if args.target == "all" else [args.target]
    for target in targets:
        deploy_one(target)


if __name__ == "__main__":
    main()

