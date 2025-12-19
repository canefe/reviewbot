from contextvars import ContextVar
from typing import Optional

from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager

store_manager_ctx: ContextVar[Optional[CodebaseStoreManager]] = ContextVar(
    "store_manager", default=None
)