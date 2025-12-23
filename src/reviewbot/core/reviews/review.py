from dataclasses import dataclass, field
from typing import List
from uuid import UUID, uuid4

from reviewbot.core.issues import Issue


@dataclass
class Review:
    id: UUID = field(default_factory=uuid4)
    repo: str = ""
    commit: str = ""
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""
