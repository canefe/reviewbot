import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests


@dataclass(frozen=True)
class FileDiff:
    old_path: Optional[str]  # None for new files
    new_path: Optional[str]  # None for deleted files
    is_new_file: bool
    is_deleted_file: bool
    is_renamed: bool
    patch: str  # full unified diff for this file


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")


def _strip_prefix(p: str) -> str:
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def _parse_paths_from_chunk(
    lines: List[str],
) -> Tuple[Optional[str], Optional[str], bool, bool, bool]:
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    is_new_file = False
    is_deleted_file = False
    is_renamed = False

    # Prefer explicit rename info if present
    rename_from = None
    rename_to = None

    for ln in lines:
        if ln.startswith("new file mode"):
            is_new_file = True
        elif ln.startswith("deleted file mode"):
            is_deleted_file = True
        elif ln.startswith("rename from "):
            is_renamed = True
            rename_from = ln[len("rename from ") :].strip()
        elif ln.startswith("rename to "):
            is_renamed = True
            rename_to = ln[len("rename to ") :].strip()

        # Unified diff markers usually appear too
        elif ln.startswith("--- "):
            p = ln[4:].strip()
            old_path = None if p == "/dev/null" else _strip_prefix(p)
        elif ln.startswith("+++ "):
            p = ln[4:].strip()
            new_path = None if p == "/dev/null" else _strip_prefix(p)

    # Fallback: use diff --git header if ---/+++ not present
    m = _DIFF_HEADER_RE.match(lines[0]) if lines else None
    if m:
        hdr_old = m.group(1)
        hdr_new = m.group(2)
        if old_path is None and not is_new_file:
            old_path = hdr_old
        if new_path is None and not is_deleted_file:
            new_path = hdr_new

    # Rename paths override everything (Git uses plain paths here, not a/b)
    if is_renamed:
        if rename_from:
            old_path = rename_from
        if rename_to:
            new_path = rename_to

    return old_path, new_path, is_new_file, is_deleted_file, is_renamed


def _split_raw_diff_by_file(raw: str) -> List[str]:
    # Split on "diff --git", keeping the header line with each chunk
    lines = raw.splitlines(keepends=True)
    chunks: List[List[str]] = []
    cur: List[str] = []

    for ln in lines:
        if ln.startswith("diff --git "):
            if cur:
                chunks.append(cur)
            cur = [ln]
        else:
            if cur:  # ignore anything before first diff header
                cur.append(ln)

    if cur:
        chunks.append(cur)

    return ["".join(c) for c in chunks]


def fetch_mr_diffs(
    api_v4: str,
    project_id: str,
    mr_iid: str,
    token: str,
    timeout: int = 30,
) -> List[FileDiff]:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}

    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    diff_url = f"{mr_url}/raw_diffs"

    r = requests.get(mr_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    r = requests.get(diff_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    raw = r.text
    file_chunks = _split_raw_diff_by_file(raw)

    out: List[FileDiff] = []
    for chunk in file_chunks:
        lines = chunk.splitlines(keepends=False)
        if not lines:
            continue
        if not lines[0].startswith("diff --git "):
            continue

        old_path, new_path, is_new_file, is_deleted_file, is_renamed = (
            _parse_paths_from_chunk(lines)
        )
        out.append(
            FileDiff(
                old_path=old_path,
                new_path=new_path,
                is_new_file=is_new_file,
                is_deleted_file=is_deleted_file,
                is_renamed=is_renamed,
                patch=chunk,
            )
        )

    return out


def get_mr_branch(
    api_v4: str, project_id: str, mr_iid: str, token: str, timeout: int = 30
) -> str:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}
    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    r = requests.get(mr_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["source_branch"]
