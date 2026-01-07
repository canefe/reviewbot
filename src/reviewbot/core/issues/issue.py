from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class IssueSeverity(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Issue:
    title: str
    description: str
    file_path: str
    start_line: int
    end_line: int
    severity: IssueSeverity
    status: str
    suggestion: str | None = None  # Optional code suggestion to fix the issue
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    discussion_id: str | None = None
    note_id: str | None = None
