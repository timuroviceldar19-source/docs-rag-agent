import subprocess
from pathlib import Path


def clone_or_pull(repo_url: str, target_dir: Path) -> None:
    """Clone repo to target_dir, or git pull if it already exists."""
    if (target_dir / ".git").exists():
        subprocess.run(["git", "-C", str(target_dir), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)


def collect_markdown_files(docs_root: Path) -> list[Path]:
    """Return sorted list of all *.md files under docs_root (recursive)."""
    return sorted(docs_root.rglob("*.md"))
