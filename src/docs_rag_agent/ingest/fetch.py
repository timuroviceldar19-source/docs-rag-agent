import subprocess
from pathlib import Path


def clone_or_pull(repo_url: str, target_dir: Path) -> None:
    """Clone repo to target_dir, or git pull if it already exists."""
    if (target_dir / ".git").exists():
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)


def collect_markdown_files(docs_root: Path) -> list[Path]:
    """Return sorted list of *.md files under docs_root, recursively.

    Files starting with an underscore (FastAPI's convention for non-content
    fixtures like ``_llm-test.md``) are skipped — they pollute retrieval
    quality without adding signal.
    """
    return sorted(p for p in docs_root.rglob("*.md") if not p.name.startswith("_"))
