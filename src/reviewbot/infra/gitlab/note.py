from typing import Any, Dict, List, Optional

import requests
from rich.console import Console

console = Console()


def post_merge_request_note(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    body: str,
    timeout: int = 30,
) -> None:
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/notes"

    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": token},
        data={"body": body},
        timeout=timeout,
    )

    if r.status_code >= 300:
        raise RuntimeError(
            f"gitlab note post failed: {r.status_code} {r.reason}: {r.text}"
        )


def post_discussion(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    body: str,
    position: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> str:
    """
    Create a new discussion and return its ID.

    Args:
        api_v4: GitLab API v4 base URL
        token: GitLab API token
        project_id: Project ID
        mr_iid: Merge request IID
        body: Discussion body content
        position: Optional position object for file-based discussions
        timeout: Request timeout

    Returns:
        The discussion ID from GitLab
    """
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/discussions"

    # Prepare request data
    # Note: GitLab requires either line_code or complete position with line numbers
    # For file-level discussions without specific lines, don't include position
    data: Dict[str, Any] = {"body": body}
    if position:
        # Only include position if it has required fields (new_line or old_line)
        # Otherwise GitLab will reject it as incomplete
        has_line_info = (
            "new_line" in position or "old_line" in position or "line_code" in position
        )
        if has_line_info:
            data["position"] = position
        else:
            # Position is incomplete, skip it for file-level discussions
            console.print(
                "[yellow]Position object missing line information, creating discussion without position[/yellow]"
            )

    # Use json parameter to send as JSON (not form data)
    # This is required for nested objects like position
    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": token},
        json=data,
        timeout=timeout,
    )

    # Log error details if request fails
    if r.status_code >= 400:
        console.print(f"[red]Request failed with status {r.status_code}[/red]")
        console.print(f"[red]Request data: {data}[/red]")
        try:
            error_response = r.json()
            console.print(f"[red]Error response: {error_response}[/red]")
        except Exception:
            console.print(f"[red]Error response text: {r.text}[/red]")

    r.raise_for_status()

    # GitLab returns the created discussion with an 'id' field
    response_data = r.json()
    discussion_id = response_data.get("id")

    if not discussion_id:
        raise RuntimeError(f"Discussion created but no ID returned: {response_data}")

    return discussion_id


def post_discussion_reply(
    api_v4: str,
    token: str,
    project_id: str,
    merge_request_id: str,
    discussion_id: str,
    body: str,
    timeout: int = 30,
) -> None:
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{merge_request_id}/discussions/{discussion_id}/notes"
    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": token},
        data={"body": body},
        timeout=timeout,
    )
    r.raise_for_status()


# Wrapper functions for easier use
def create_discussion(
    title: str,
    body: str,
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
) -> str:
    """
    Create a discussion with title and body.

    Returns:
        Discussion ID
    """
    # GitLab discussions don't have separate titles, so we include it in the body
    full_body = f"## {title}\n\n{body}"

    discussion_id = post_discussion(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        mr_iid=mr_iid,
        body=full_body,
    )

    return discussion_id


def reply_to_discussion(
    discussion_id: str,
    body: str,
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
) -> None:
    """
    Reply to an existing discussion.
    """
    post_discussion_reply(
        api_v4=api_v4,
        token=token,
        project_id=project_id,
        merge_request_id=mr_iid,
        discussion_id=discussion_id,
        body=body,
    )


def delete_discussion(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    discussion_id: str,
    note_id: str,
    timeout: int = 30,
) -> None:
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/discussions/{discussion_id}/notes/{note_id}"
    r = requests.delete(
        url,
        headers={"PRIVATE-TOKEN": token},
        timeout=timeout,
    )
    r.raise_for_status()


def get_all_discussions(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/discussions"
    r = requests.get(
        url,
        headers={"PRIVATE-TOKEN": token},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def get_merge_request_notes(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Get all notes (comments) for a merge request.

    Args:
        api_v4: GitLab API v4 base URL
        token: GitLab API token
        project_id: Project ID
        mr_iid: Merge request IID
        timeout: Request timeout

    Returns:
        List of note dictionaries from GitLab API
    """
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/notes"
    r = requests.get(
        url,
        headers={"PRIVATE-TOKEN": token},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()
