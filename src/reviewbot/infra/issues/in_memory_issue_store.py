from collections.abc import Iterable
from uuid import UUID

from langgraph.store.memory import InMemoryStore

from reviewbot.core.issues.issue import Issue
from reviewbot.core.issues.issue_model import IssueModel
from reviewbot.core.issues.issue_store import IssueStore


class InMemoryIssueStore(IssueStore):
    NS = ("issues",)

    def __init__(self, store: InMemoryStore):
        self.store = store

    def add(self, issue: Issue) -> None:
        model = IssueModel.from_domain(issue)
        self.store.put(self.NS, str(issue.id), model.model_dump())

    def get(self, issue_id: UUID) -> Issue | None:
        raw = self.store.get(self.NS, str(issue_id))
        if raw is None:
            return None
        return IssueModel.model_validate(raw).to_domain()

    def list(self) -> Iterable[Issue]:
        items: list[Issue] = []
        for raw in self.store.search(self.NS):
            items.append(IssueModel.model_validate(raw).to_domain())
        return items

    def update(self, issue: Issue) -> None:
        self.add(issue)

    def delete(self, issue_id: UUID) -> None:
        self.store.delete(self.NS, str(issue_id))
