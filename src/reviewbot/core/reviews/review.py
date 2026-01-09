from dataclasses import dataclass, field
from uuid import UUID, uuid4

from reviewbot.core.issues import Issue


@dataclass
class Review:
    id: UUID = field(default_factory=uuid4)
    repo: str = ""
    commit: str = ""
    issues: list[Issue] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class Acknowledgment:
    discussion_id: str
    note_id: str
    created_at: str
