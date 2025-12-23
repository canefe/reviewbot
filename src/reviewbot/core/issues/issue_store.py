from abc import ABC, abstractmethod
from typing import Iterable, Optional
from uuid import UUID

from .issue import Issue


class IssueStore(ABC):
    @abstractmethod
    def add(self, issue: Issue) -> None: ...

    @abstractmethod
    def get(self, issue_id: UUID) -> Optional[Issue]: ...

    @abstractmethod
    def list(self) -> Iterable[Issue]: ...

    @abstractmethod
    def update(self, issue: Issue) -> None: ...

    @abstractmethod
    def delete(self, issue_id: UUID) -> None: ...
