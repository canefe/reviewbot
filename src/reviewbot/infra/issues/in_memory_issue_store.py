from typing import Dict, Iterable, Optional
from uuid import UUID

from reviewbot.core.issues.issue import Issue
from reviewbot.core.issues.issue_store import IssueStore


class InMemoryIssueStore(IssueStore):
    def __init__(self) -> None:
        self._items: Dict[UUID, Issue] = {}

    def add(self, issue: Issue) -> None:
        self._items[issue.id] = issue

    def get(self, issue_id: UUID) -> Optional[Issue]:
        return self._items.get(issue_id)

    def list(self) -> Iterable[Issue]:
        return self._items.values()

    def update(self, issue: Issue) -> None:
        self._items[issue.id] = issue

    def delete(self, issue_id: UUID) -> None:
        self._items.pop(issue_id, None)
