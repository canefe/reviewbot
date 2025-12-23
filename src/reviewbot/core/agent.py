from typing import Any

from langgraph.graph.state import CompiledStateGraph  # type: ignore

Agent = CompiledStateGraph[Any, Any, Any, Any]
