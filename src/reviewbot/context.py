from contextvars import ContextVar
from typing import TypedDict

from reviewbot.core.issues import IssueStore
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager


class Context(TypedDict):
    store_manager: CodebaseStoreManager | None
    issue_store: IssueStore | None


store_manager_ctx: ContextVar[Context | None] = ContextVar("store_manager", default=None)

def init_ctx() -> None:
    store_manager_ctx.set({"store_manager": None, "issue_store": None})
