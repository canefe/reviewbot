import hashlib
import re
from typing import Any


def _extract_code_from_diff(diff_text: str, line_start: int, line_end: int) -> str:
    hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    lines = diff_text.splitlines()

    extracted = []
    current_new = 0
    in_hunk = False

    for line in lines:
        match = hunk_header_pattern.match(line)
        if match:
            current_new = int(match.group(3))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # We only care about the lines in the NEW file (the result of the change)
        if line.startswith("+"):
            if line_start <= current_new <= line_end:
                extracted.append(line[1:])  # Remove '+'
            current_new += 1
        elif line.startswith("-"):
            # Skip deleted lines for code extraction of the 'new' state
            continue
        else:
            # Context line
            if line_start <= current_new <= line_end:
                extracted.append(line[1:] if line else "")
            current_new += 1

        # FIX: Exit early if we've passed the end of our requested range
        if current_new > line_end:
            if extracted:  # Only break if we actually found lines
                break

    return "\n".join(extracted)


def generate_line_code(file_path: str, old_line: int | None, new_line: int | None) -> str:
    """
    Generates a GitLab-compatible line_code.
    Format: sha1(path) + "_" + old_line + "_" + new_line
    """
    path_hash = hashlib.sha1(file_path.encode()).hexdigest()
    old_s = str(old_line) if old_line is not None else ""
    new_s = str(new_line) if new_line is not None else ""
    return f"{path_hash}_{old_s}_{new_s}"


def create_position_for_issue(
    diff_text: str,
    issue_line_start: int,
    issue_line_end: int,
    base_sha: str,
    head_sha: str,
    start_sha: str,
    old_path: str,
    new_path: str,
) -> dict[str, Any] | None:
    hunk_header_pattern = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")
    lines = diff_text.splitlines()

    current_old, current_new = 0, 0
    in_hunk = False

    # Track the actual lines found in the diff to build the range
    matched_lines = []  # List of (old_line, new_line)

    for line in lines:
        match = hunk_header_pattern.match(line)
        if match:
            current_old, current_new = int(match.group(1)), int(match.group(3))
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # Logic to determine if this specific line is within our target range
        if line.startswith("+"):
            if issue_line_start <= current_new <= issue_line_end:
                matched_lines.append((None, current_new))
            current_new += 1
        elif line.startswith("-"):
            if issue_line_start <= current_old <= issue_line_end:
                matched_lines.append((current_old, None))
            current_old += 1
        else:
            if issue_line_start <= current_new <= issue_line_end:
                matched_lines.append((current_old, current_new))
            current_old += 1
            current_new += 1

        # FIX: Optimization to prevent "sticky" hunk matching.
        # If we have passed the end_line in the NEW file, we stop.
        if current_new > issue_line_end and not line.startswith("-"):
            if matched_lines:
                break

    if not matched_lines:
        return None

    # We anchor the comment to the LAST line of the range so the code is visible
    start_old, start_new = matched_lines[0]
    end_old, end_new = matched_lines[-1]

    # Calculate line codes for the range
    start_code = generate_line_code(new_path if start_new else old_path, start_old, start_new)
    end_code = generate_line_code(new_path if end_new else old_path, end_old, end_new)

    position = {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "start_sha": start_sha,
        "position_type": "text",
        "old_path": old_path,
        "new_path": new_path,
        # Anchor the main comment on the end of the range
        "new_line": end_new,
        "old_line": end_old,
        "line_range": {
            "start": {
                "line_code": start_code,
                "type": "new" if start_new else "old",
                "new_line": start_new,
                "old_line": start_old,
            },
            "end": {
                "line_code": end_code,
                "type": "new" if end_new else "old",
                "new_line": end_new,
                "old_line": end_old,
            },
        },
    }

    # Cleanup: GitLab doesn't like None values in the schema
    if position["new_line"] is None:
        del position["new_line"]
    if position["old_line"] is None:
        del position["old_line"]

    return position


def create_file_position(
    base_sha: str,
    head_sha: str,
    start_sha: str,
    old_path: str,
    new_path: str,
) -> dict[str, Any]:
    return {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "start_sha": start_sha,
        "position_type": "file",
        "old_path": old_path,
        "new_path": new_path,
    }
