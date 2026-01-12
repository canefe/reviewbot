from typing import Any

from langchain.tools import ToolRuntime, tool  # type: ignore

from reviewbot.core.issues import Issue, IssueSeverity
from reviewbot.core.issues.issue_model import IssueModel


@tool
def add_issue(
    title: str,
    description: str,
    file_path: str,
    start_line: int,
    end_line: int,
    severity: IssueSeverity,
    status: str,
    runtime: ToolRuntime[None, dict[str, Any]],
) -> str:
    """Add an issue to the issue store.

    Args:
        title: title of the issue
        description: description of the issue
        file_path: path to the file that contains the issue
        start_line: start line number of the issue
        end_line: end line number of the issue
        severity: severity of the issue
        status: status of the issue

    Returns:
        string with the id of the added issue
    """
    if not runtime.store:
        raise ValueError("Store not found in runtime")

    issue = Issue(
        title=title,
        description=description,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        severity=severity,
        status=status,
    )

    issue_model = IssueModel.from_domain(issue)

    runtime.store.put(("issues",), str(issue.id), issue_model.model_dump())
    return f"Issue added successfully: {issue.id}"
