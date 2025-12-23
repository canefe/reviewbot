from langchain.tools import tool  # type: ignore

from reviewbot.context import store_manager_ctx
from reviewbot.core.issues import Issue, IssueSeverity


@tool
def add_issue(
    title: str,
    description: str,
    file_path: str,
    line_number: int,
    column_number: int,
    severity: IssueSeverity,
    status: str,
) -> str:
    """Add an issue to the issue store.

    Args:
        title: title of the issue
        description: description of the issue
        file_path: path to the file that contains the issue
        line_number: line number of the issue
        column_number: column number of the issue
        severity: severity of the issue
        status: status of the issue

    Returns:
        string with the id of the added issue
    """
    context = store_manager_ctx.get()
    issue_store = context.get("issue_store")
    if not issue_store:
        return "Issue store not found."

    issue = Issue(
        title=title,
        description=description,
        file_path=file_path,
        line_number=line_number,
        column_number=column_number,
        severity=severity,
        status=status,
    )

    issue_store.add(issue)
    return f"Issue added successfully: {issue.id}"
