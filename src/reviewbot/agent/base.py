from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langgraph.func import entrypoint  # type: ignore
from rich.console import Console

from reviewbot.agent.tasks.core import ToolCallerSettings
from reviewbot.agent.tasks.issues import IssuesInput, identify_issues
from reviewbot.context import Context
from reviewbot.core.agent import Agent
from reviewbot.core.issues import Issue, IssueModel

console = Console()


# Generate response workflow model
@dataclass
class AgentRunnerInput:
    agent: Agent
    context: Context
    settings: ToolCallerSettings = field(default_factory=ToolCallerSettings)
    on_file_complete: Optional[Callable[[str, List[IssueModel]], None]] = None
    quick_scan_agent: Optional[Agent] = None


@entrypoint()
def agent_runner(input: AgentRunnerInput) -> List[Issue]:
    agent = input.agent
    settings = input.settings
    context = input.context
    on_file_complete = input.on_file_complete
    quick_scan_agent = input.quick_scan_agent

    issue_store = context.get("issue_store")
    if not issue_store:
        raise ValueError("Issue store not found")

    store_manager = context.get("store_manager")
    if not store_manager:
        raise ValueError("Store manager not found")

    # Step 1: Identify the issues
    issues = identify_issues(
        ctx=IssuesInput(
            agent=agent,
            context=context,
            settings=settings,
            on_file_complete=on_file_complete,
            quick_scan_agent=quick_scan_agent,
        )
    ).result()

    return issues
