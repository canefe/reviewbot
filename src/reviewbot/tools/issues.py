from langchain.tools import ToolRuntime, tool  # type: ignore

from reviewbot.core.issues import Issue, IssueSeverity
from reviewbot.infra.issues.in_memory_issue_store import InMemoryIssueStore


@tool
def add_issue(
    title: str,
    description: str,
    file_path: str,
    start_line: int,
    end_line: int,
    severity: IssueSeverity,
    status: str,
    runtime: ToolRuntime,
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
    issue_store = InMemoryIssueStore(runtime.store)

    issue = Issue(
        title=title,
        description=description,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        severity=severity,
        status=status,
    )

    issue_store.add(issue)
    return f"Issue added successfully: {issue.id}"
