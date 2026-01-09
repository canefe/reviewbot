import json
import re
from typing import Any, Literal

import httpx
import requests
from pydantic import BaseModel
from rich.console import Console

console = Console()


class LineRangePoint(BaseModel):
    line_code: str
    type: Literal["old", "new"]
    old_line: int | None = None
    new_line: int | None = None

    model_config = {"extra": "forbid", "frozen": True}


class LineRange(BaseModel):
    start: LineRangePoint
    end: LineRangePoint

    model_config = {"extra": "forbid", "frozen": True}


class DiffPosition(BaseModel):
    base_sha: str
    start_sha: str
    head_sha: str

    position_type: Literal["text", "image", "file"]

    old_path: str | None = None
    new_path: str | None = None

    old_line: int | None = None
    new_line: int | None = None

    line_range: LineRange | None = None

    # image-only fields
    width: int | None = None
    height: int | None = None
    x: float | None = None
    y: float | None = None

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }


class FileDiff(BaseModel):
    old_path: str | None
    new_path: str | None
    is_new_file: bool
    is_deleted_file: bool
    is_renamed: bool
    patch: str
    position: DiffPosition | None = None

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")


def _strip_prefix(p: str) -> str:
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def _parse_paths_from_chunk(
    lines: list[str],
) -> tuple[str | None, str | None, bool, bool, bool]:
    old_path: str | None = None
    new_path: str | None = None
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


def _split_raw_diff_by_file(raw: str) -> list[str]:
    # Split on "diff --git", keeping the header line with each chunk
    lines = raw.splitlines(keepends=True)
    chunks: list[list[str]] = []
    cur: list[str] = []

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
) -> tuple[list[FileDiff], dict[str, str]]:
    """
    Fetch merge request diffs from GitLab API.

    Supports both the old raw diff format and the new JSON changes format.
    The new format includes position information for discussions.
    """
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}

    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    changes_url = f"{mr_url}/changes"

    # Get merge request info to extract diff_refs for position objects
    mr_response = requests.get(mr_url, headers=headers, timeout=timeout)
    mr_response.raise_for_status()
    mr_data = mr_response.json()

    # Get diff_refs for position objects
    diff_refs = mr_data.get("diff_refs") or {}
    base_sha = diff_refs.get("base_sha")
    head_sha = diff_refs.get("head_sha")
    start_sha = diff_refs.get("start_sha")
    mr_web_url = mr_data.get("web_url")
    if mr_web_url and "/-/merge_requests/" in mr_web_url:
        diff_refs["project_web_url"] = mr_web_url.split("/-/merge_requests/")[0]

    # Try the new JSON changes endpoint first
    changes_response = requests.get(changes_url, headers=headers, timeout=timeout)
    changes_response.raise_for_status()

    try:
        # Try to parse as JSON (new format)
        changes_data = changes_response.json()

        if isinstance(changes_data, dict) and "changes" in changes_data:
            # New JSON format with changes array
            file_diffs: list[FileDiff] = []

            for change in changes_data["changes"]:
                change_old_path: str | None = change.get("old_path")
                change_new_path: str | None = change.get("new_path")
                diff_text: str = change.get("diff", "")
                change_is_new_file: bool = change.get("new_file", False)
                change_is_deleted_file: bool = change.get("deleted_file", False)
                change_is_renamed: bool = change.get("renamed_file", False)

                # Create position object for discussions
                change_position: dict[str, Any] | None = None
                if base_sha and head_sha and start_sha:
                    # Parse diff to find first hunk with line range information
                    # Parse diff to find first hunk
                    # Parse diff to find first changed line
                    hunk_header_pattern = re.compile(
                        r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
                    )

                    change_old_line: int | None = None
                    change_new_line: int | None = None

                    lines = diff_text.splitlines()
                    in_hunk = False
                    current_old = 0
                    current_new = 0

                    for line in lines:
                        # Check for hunk header
                        match = hunk_header_pattern.match(line)
                        if match:
                            current_old = int(match.group(1))
                            current_new = int(match.group(3))
                            in_hunk = True
                            continue

                        if not in_hunk:
                            continue

                        # Found a change - use this line!
                        if line.startswith("-"):
                            change_old_line = current_old
                            change_new_line = None  # Deletion has no new line
                            break
                        elif line.startswith("+"):
                            change_old_line = None  # Addition has no old line
                            change_new_line = current_new
                            break

                        # Context line - increment counters
                        if line.startswith(" ") or (
                            line and not line.startswith(("@@", "\\", "diff"))
                        ):
                            current_old += 1
                            current_new += 1

                    # Create position object
                    change_position = {
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "start_sha": start_sha,
                        "old_path": change_old_path,
                        "new_path": change_new_path,
                        "position_type": "text",
                    }

                    if change_new_line is not None:
                        change_position["new_line"] = change_new_line

                    if change_old_line is not None:
                        change_position["old_line"] = change_old_line

                    # Default fallback
                    if change_new_line is None and change_old_line is None:
                        change_position["new_line"] = 1
                    # if line

                # If diff is empty or too large, try to get it from raw_diffs endpoint
                if not diff_text or change.get("too_large", False):
                    # Fallback to raw diff endpoint for this file
                    raw_diff_url = f"{mr_url}/diffs"
                    raw_response = requests.get(raw_diff_url, headers=headers, timeout=timeout)
                    raw_response.raise_for_status()
                    raw_diff = raw_response.text

                    # Extract this file's diff from raw format
                    file_chunks = _split_raw_diff_by_file(raw_diff)
                    for chunk in file_chunks:
                        lines = chunk.splitlines(keepends=False)
                        if not lines or not lines[0].startswith("diff --git "):
                            continue

                        chunk_old_path, chunk_new_path, _, _, _ = _parse_paths_from_chunk(lines)
                        if (chunk_new_path == change_new_path) or (
                            chunk_old_path == change_old_path
                        ):
                            diff_text = chunk
                            break

                file_diffs.append(
                    FileDiff(
                        old_path=change_old_path,
                        new_path=change_new_path,
                        is_new_file=change_is_new_file,
                        is_deleted_file=change_is_deleted_file,
                        is_renamed=change_is_renamed,
                        patch=diff_text,
                        position=None,
                    )
                )

            return file_diffs, diff_refs

    except (json.JSONDecodeError, KeyError):
        # Fallback to old raw diff format
        pass

    # Old format: parse raw diff text
    raw_diff_url = f"{mr_url}/diffs"
    raw_response = requests.get(raw_diff_url, headers=headers, timeout=timeout)
    raw_response.raise_for_status()
    raw = raw_response.text

    file_chunks = _split_raw_diff_by_file(raw)

    out: list[FileDiff] = []
    for chunk in file_chunks:
        lines = chunk.splitlines(keepends=False)
        if not lines:
            continue
        if not lines[0].startswith("diff --git "):
            continue

        (
            parsed_old_path,
            parsed_new_path,
            parsed_is_new_file,
            parsed_is_deleted_file,
            parsed_is_renamed,
        ) = _parse_paths_from_chunk(lines)

        # Create position object for discussions
        # GitLab requires line_code or line numbers (new_line/old_line)
        # Extract the first line number from the diff for file-level positioning
        raw_position: dict[str, Any] | None = None
        if base_sha and head_sha and start_sha:
            # Try to extract the first line number from the diff
            extracted_new_line: int | None = None
            extracted_old_line: int | None = None

            # Parse diff to find first hunk and line numbers
            hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
            for diff_line in chunk.splitlines():
                match = hunk_header_pattern.match(diff_line)
                if match:
                    extracted_old_line = int(match.group(1))
                    extracted_new_line = int(match.group(3))
                    break

            # Create position object with line information
            raw_position = {
                "base_sha": base_sha,
                "head_sha": head_sha,
                "start_sha": start_sha,
                "old_path": parsed_old_path,
                "new_path": parsed_new_path,
                "position_type": "text",
            }

            # Add line numbers if we found them
            if extracted_new_line is not None:
                raw_position["new_line"] = extracted_new_line
            if extracted_old_line is not None:
                raw_position["old_line"] = extracted_old_line

            # If no lines found, use line 1 as default for file-level discussion
            if extracted_new_line is None and extracted_old_line is None:
                raw_position["new_line"] = 1

        out.append(
            FileDiff(
                old_path=parsed_old_path,
                new_path=parsed_new_path,
                is_new_file=parsed_is_new_file,
                is_deleted_file=parsed_is_deleted_file,
                is_renamed=parsed_is_renamed,
                patch=chunk,
                position=raw_position,
            )
        )

    return out, diff_refs


def get_mr_branch(api_v4: str, project_id: str, mr_iid: str, token: str, timeout: int = 30) -> str:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}
    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    r = requests.get(mr_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["source_branch"]


# ============================================================================
# ASYNC VERSIONS (using httpx)
# ============================================================================


async def async_fetch_mr_diffs(
    api_v4: str,
    project_id: str,
    mr_iid: str,
    token: str,
    timeout: int = 30,
) -> tuple[list[FileDiff], dict[str, str]]:
    """
    Fetch merge request diffs from GitLab API (async version).

    Supports both the old raw diff format and the new JSON changes format.
    The new format includes position information for discussions.
    """
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}

    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    changes_url = f"{mr_url}/changes"

    async with httpx.AsyncClient() as client:
        # Get merge request info to extract diff_refs for position objects
        mr_response = await client.get(mr_url, headers=headers, timeout=timeout)
        mr_response.raise_for_status()
        mr_data = mr_response.json()

        # Get diff_refs for position objects
        diff_refs = mr_data.get("diff_refs") or {}
        base_sha = diff_refs.get("base_sha")
        head_sha = diff_refs.get("head_sha")
        start_sha = diff_refs.get("start_sha")
        mr_web_url = mr_data.get("web_url")
        if mr_web_url and "/-/merge_requests/" in mr_web_url:
            diff_refs["project_web_url"] = mr_web_url.split("/-/merge_requests/")[0]

        # Try the new JSON changes endpoint first
        changes_response = await client.get(changes_url, headers=headers, timeout=timeout)
        changes_response.raise_for_status()

        try:
            # Try to parse as JSON (new format)
            changes_data = changes_response.json()

            if isinstance(changes_data, dict) and "changes" in changes_data:
                # New JSON format with changes array
                file_diffs: list[FileDiff] = []

                for change in changes_data["changes"]:
                    change_old_path: str | None = change.get("old_path")
                    change_new_path: str | None = change.get("new_path")
                    diff_text: str = change.get("diff", "")
                    change_is_new_file: bool = change.get("new_file", False)
                    change_is_deleted_file: bool = change.get("deleted_file", False)
                    change_is_renamed: bool = change.get("renamed_file", False)

                    # Create position object for discussions
                    change_position: dict[str, Any] | None = None
                    if base_sha and head_sha and start_sha:
                        # Parse diff to find first hunk with line range information
                        hunk_header_pattern = re.compile(
                            r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"
                        )

                        change_old_line: int | None = None
                        change_new_line: int | None = None

                        lines = diff_text.splitlines()
                        in_hunk = False
                        current_old = 0
                        current_new = 0

                        for line in lines:
                            # Check for hunk header
                            match = hunk_header_pattern.match(line)
                            if match:
                                current_old = int(match.group(1))
                                current_new = int(match.group(3))
                                in_hunk = True
                                continue

                            if not in_hunk:
                                continue

                            # Found a change - use this line!
                            if line.startswith("-"):
                                change_old_line = current_old
                                change_new_line = None  # Deletion has no new line
                                break
                            elif line.startswith("+"):
                                change_old_line = None  # Addition has no old line
                                change_new_line = current_new
                                break

                            # Context line - increment counters
                            if line.startswith(" ") or (
                                line and not line.startswith(("@@", "\\", "diff"))
                            ):
                                current_old += 1
                                current_new += 1

                        # Create position object
                        change_position = {
                            "base_sha": base_sha,
                            "head_sha": head_sha,
                            "start_sha": start_sha,
                            "old_path": change_old_path,
                            "new_path": change_new_path,
                            "position_type": "text",
                        }

                        if change_new_line is not None:
                            change_position["new_line"] = change_new_line

                        if change_old_line is not None:
                            change_position["old_line"] = change_old_line

                        # Default fallback
                        if change_new_line is None and change_old_line is None:
                            change_position["new_line"] = 1

                    # If diff is empty or too large, try to get it from raw_diffs endpoint
                    if not diff_text or change.get("too_large", False):
                        # Fallback to raw diff endpoint for this file
                        raw_diff_url = f"{mr_url}/diffs"
                        raw_response = await client.get(
                            raw_diff_url, headers=headers, timeout=timeout
                        )
                        raw_response.raise_for_status()
                        raw_diff = raw_response.text

                        # Extract this file's diff from raw format
                        file_chunks = _split_raw_diff_by_file(raw_diff)
                        for chunk in file_chunks:
                            lines = chunk.splitlines(keepends=False)
                            if not lines or not lines[0].startswith("diff --git "):
                                continue

                            chunk_old_path, chunk_new_path, _, _, _ = _parse_paths_from_chunk(lines)
                            if (chunk_new_path == change_new_path) or (
                                chunk_old_path == change_old_path
                            ):
                                diff_text = chunk
                                break

                    file_diffs.append(
                        FileDiff(
                            old_path=change_old_path,
                            new_path=change_new_path,
                            is_new_file=change_is_new_file,
                            is_deleted_file=change_is_deleted_file,
                            is_renamed=change_is_renamed,
                            patch=diff_text,
                            position=None,
                        )
                    )

                return file_diffs, diff_refs

        except (json.JSONDecodeError, KeyError):
            # Fallback to old raw diff format
            pass

        # Old format: parse raw diff text
        raw_diff_url = f"{mr_url}/diffs"
        raw_response = await client.get(raw_diff_url, headers=headers, timeout=timeout)
        raw_response.raise_for_status()
        raw = raw_response.text

    file_chunks = _split_raw_diff_by_file(raw)

    out: list[FileDiff] = []
    for chunk in file_chunks:
        lines = chunk.splitlines(keepends=False)
        if not lines:
            continue
        if not lines[0].startswith("diff --git "):
            continue

        (
            parsed_old_path,
            parsed_new_path,
            parsed_is_new_file,
            parsed_is_deleted_file,
            parsed_is_renamed,
        ) = _parse_paths_from_chunk(lines)

        # Create position object for discussions
        raw_position: dict[str, Any] | None = None
        if base_sha and head_sha and start_sha:
            extracted_new_line: int | None = None
            extracted_old_line: int | None = None

            # Parse diff to find first hunk and line numbers
            hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
            for diff_line in chunk.splitlines():
                match = hunk_header_pattern.match(diff_line)
                if match:
                    extracted_old_line = int(match.group(1))
                    extracted_new_line = int(match.group(3))
                    break

            # Create position object with line information
            raw_position = {
                "base_sha": base_sha,
                "head_sha": head_sha,
                "start_sha": start_sha,
                "old_path": parsed_old_path,
                "new_path": parsed_new_path,
                "position_type": "text",
            }

            # Add line numbers if we found them
            if extracted_new_line is not None:
                raw_position["new_line"] = extracted_new_line
            if extracted_old_line is not None:
                raw_position["old_line"] = extracted_old_line

            # If no lines found, use line 1 as default for file-level discussion
            if extracted_new_line is None and extracted_old_line is None:
                raw_position["new_line"] = 1

        out.append(
            FileDiff(
                old_path=parsed_old_path,
                new_path=parsed_new_path,
                is_new_file=parsed_is_new_file,
                is_deleted_file=parsed_is_deleted_file,
                is_renamed=parsed_is_renamed,
                patch=chunk,
                position=raw_position,
            )
        )

    return out, diff_refs


async def async_get_mr_branch(
    api_v4: str, project_id: str, mr_iid: str, token: str, timeout: int = 30
) -> str:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}
    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"

    async with httpx.AsyncClient() as client:
        r = await client.get(mr_url, headers=headers, timeout=timeout)

    r.raise_for_status()
    return r.json()["source_branch"]
