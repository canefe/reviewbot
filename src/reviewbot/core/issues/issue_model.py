from pydantic import BaseModel, ConfigDict

from reviewbot.core.issues.issue import Issue, IssueSeverity


class IssueModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str
    description: str
    file_path: str
    line_number: int
    column_number: int
    severity: IssueSeverity
    status: str

    def to_domain(self) -> Issue:
        return Issue(**self.model_dump())
