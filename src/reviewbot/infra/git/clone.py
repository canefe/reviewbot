import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


class GitCloneError(RuntimeError):
    pass


def clone_repo_tmp(
    repo_url: str, *, branch: str | None = None
) -> TemporaryDirectory[str]:
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
