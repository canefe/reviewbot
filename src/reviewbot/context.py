from contextvars import ContextVar
from typing import Optional, TypedDict

from reviewbot.core.issues import IssueStore
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager


class Context(TypedDict):
    store_manager: Optional[CodebaseStoreManager]
    issue_store: Optional[IssueStore]


store_manager_ctx: ContextVar[Context] = ContextVar(
    "store_manager", default=Context(store_manager=None, issue_store=None)
)
