from dataclasses import dataclass, field
from typing import List

from langgraph.func import entrypoint  # type: ignore
from rich.console import Console

from reviewbot.agent.tasks.core import ToolCallerSettings
from reviewbot.agent.tasks.issues import IssuesInput, identify_issues
from reviewbot.context import Context
from reviewbot.core.agent import Agent
from reviewbot.core.issues import Issue

console = Console()


# Generate response workflow model
@dataclass
class AgentRunnerInput:
    agent: Agent
    context: Context
    settings: ToolCallerSettings = field(default_factory=ToolCallerSettings)


@entrypoint()
def agent_runner(input: AgentRunnerInput) -> List[Issue]:
    agent = input.agent
    settings = input.settings
    context = input.context

    issue_store = context.get("issue_store")
    if not issue_store:
        raise ValueError("Issue store not found")

    store_manager = context.get("store_manager")
    if not store_manager:
        raise ValueError("Store manager not found")

    # Step 1: Identify the issues
    issues = identify_issues(
        ctx=IssuesInput(agent=agent, context=context, settings=settings)
    ).result()

    return issues
