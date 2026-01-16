import fnmatch
from pathlib import Path

from rich.console import Console  # type: ignore

from reviewbot.infra.gitlab.diff import FileDiff

console = Console()

# Global blacklist for common files that typically don't need code review
GLOBAL_REVIEW_BLACKLIST = [
    # Dependency management files
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "Pipfile.lock",
    "poetry.lock",
    "composer.lock",
    "go.sum",
    "go.mod",
    "Cargo.lock",
    # Build and distribution files
    "*.min.js",
    "*.min.css",
    "*.map",
    "dist/*",
    "build/*",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.exe",
    "*.o",
    "*.a",
    # Generated files
    "*.generated.*",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "*.pb.go",
    # Documentation and assets
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    # IDE and editor files
    ".vscode/*",
    ".idea/*",
    "*.swp",
    "*.swo",
    "*~",
]


def parse_reviewignore(repo_path: Path) -> list[str]:
    """
    Parse .reviewignore file from the repository.

    Args:
        repo_path: Path to the repository root

    Returns:
        List of glob patterns to ignore
    """
    reviewignore_path = repo_path / ".reviewignore"
    patterns: list[str] = []

    if not reviewignore_path.exists():
        console.print("[dim].reviewignore file not found, using global blacklist only[/dim]")
        return patterns

    try:
        with open(reviewignore_path, encoding="utf-8") as f:
            for line in f:
                # Strip whitespace
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)

        console.print(f"[dim]Loaded {len(patterns)} patterns from .reviewignore[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to read .reviewignore: {e}[/yellow]")

    return patterns


def should_ignore_file(file_path: str, reviewignore_patterns: list[str]) -> bool:
    """
    Check if a file should be ignored based on .reviewignore patterns and global blacklist.

    Args:
        file_path: Path to the file (relative to repo root)
        reviewignore_patterns: Patterns from .reviewignore file

    Returns:
        True if the file should be ignored, False otherwise
    """
    # Normalize the file path (remove leading ./ or /)
    normalized_path = file_path.lstrip("./")

    # Check against global blacklist
    for pattern in GLOBAL_REVIEW_BLACKLIST:
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Also check just the filename for non-path patterns
        if "/" not in pattern and fnmatch.fnmatch(Path(normalized_path).name, pattern):
            return True

    # Check against .reviewignore patterns
    for pattern in reviewignore_patterns:
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Also check just the filename for non-path patterns
        if "/" not in pattern and fnmatch.fnmatch(Path(normalized_path).name, pattern):
            return True

    return False


def filter_diffs(diffs: list[FileDiff], reviewignore_patterns: list[str]) -> list[FileDiff]:
    """
    Filter out diffs for files that should be ignored.

    Args:
        diffs: List of file diffs
        reviewignore_patterns: Patterns from .reviewignore file

    Returns:
        Filtered list of diffs
    """
    filtered: list[FileDiff] = []
    ignored_count = 0

    for diff in diffs:
        # Use new_path if available, otherwise use old_path
        file_path = diff.new_path or diff.old_path

        if file_path and should_ignore_file(file_path, reviewignore_patterns):
            console.print(f"[dim]⊘ Ignoring {file_path}[/dim]")
            ignored_count += 1
        else:
            filtered.append(diff)

    if ignored_count > 0:
        console.print(f"[cyan]Filtered out {ignored_count} file(s) based on ignore patterns[/cyan]")

    return filtered
