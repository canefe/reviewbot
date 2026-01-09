from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
from langgraph.store.memory import InMemoryStore  # type: ignore
from pydantic import BaseModel

from reviewbot.infra.gitlab.diff import FileDiff

# Shared store and checkpointer instances
checkpointer = InMemorySaver()
store = InMemoryStore()  # An instance of InMemoryStore for long-term memory


class CodebaseState(BaseModel):
    repo_root: str
    repo_name: str
    repo_tree: str
    diffs: list[FileDiff]

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }
