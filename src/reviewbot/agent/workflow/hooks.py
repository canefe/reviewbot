from typing import Any

from langchain.agents.middleware import (  # type: ignore
    AgentState,
    before_agent,
    before_model,
)
from langgraph.pregel.main import Runtime  # type: ignore
from rich.console import Console  # type: ignore

console = Console()


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    messages = state["messages"]
    console.print("[blue]Before modelMessages:[/blue]")
    console.print(messages[-5:])
    console.print("[blue]Before model messages end.[/blue]")
    return None


@before_agent(can_jump_to=["end"])
def check_agent_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore
    messages = state["messages"]
    console.print("[red]Before agent messages:[/red]")
    console.print(messages[-5:])
    console.print("[red]Before agent messages end.[/red]")
    return None
