from pydantic import BaseModel, ConfigDict, RootModel

from reviewbot.core.issues.issue import Issue, IssueSeverity


class IssueModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str
    description: str
    file_path: str
    start_line: int
    end_line: int
    severity: IssueSeverity
    status: str
    suggestion: str | None = None  # Optional code suggestion to fix the issue
    discussion_id: str | None = None
    note_id: str | None = None

    def to_domain(self) -> Issue:
        return Issue(**self.model_dump())


class IssueModelList(RootModel[list[IssueModel]]):
    """Wrapper for a list of IssueModel objects.

    Use .root to access the underlying list.
    """

    pass
