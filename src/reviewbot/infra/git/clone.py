import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib


class GitCloneError(RuntimeError):
    pass


def clone_repo_tmp(repo_url: str, *, branch: str | None = None) -> TemporaryDirectory[str]:
    tmp = TemporaryDirectory(prefix="reviewbot-")
    dest = Path(tmp.name)

    cmd = ["git", "clone", repo_url, str(dest)]
    if branch:
        cmd += ["--branch", branch]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        tmp.cleanup()
        raise GitCloneError(e.stderr.strip() or e.stdout.strip()) from e

    return tmp


def _repo_key(repo_url: str, branch: str | None) -> str:
    key = f"{repo_url}::{branch or 'default'}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


BASE_REPO_DIR = Path.home() / "reviewbot" / "repos"


def clone_repo_persistent(
    repo_url: str,
    *,
    branch: str | None = None,
) -> Path:
    BASE_REPO_DIR.mkdir(parents=True, exist_ok=True)

    repo_id = _repo_key(repo_url, branch)
    repo_dir = BASE_REPO_DIR / repo_id

    if repo_dir.exists():
        # ensure it's up to date
        subprocess.run(
            ["git", "fetch", "--all", "--prune"],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if branch:
            subprocess.run(
                ["git", "checkout", branch],
                cwd=repo_dir,
                check=True,
            )
            subprocess.run(
                ["git", "reset", "--hard", f"origin/{branch}"],
                cwd=repo_dir,
                check=True,
            )
        return repo_dir

    cmd = ["git", "clone"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [repo_url, str(repo_dir)]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise GitCloneError(e.stderr.strip() or e.stdout.strip()) from e

    return repo_dir


def get_repo_name(repo_dir: Path) -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    result = subprocess.run(
        cmd,
        check=True,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()
