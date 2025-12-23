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
    timeout: int = 30,
) -> str:
    """
    Create a new discussion and return its ID.

    Returns:
        The discussion ID from GitLab
    """
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/discussions"
    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": token},
        data={"body": body},
        timeout=timeout,
    )
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
