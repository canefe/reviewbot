from langchain.tools import tool  # type: ignore

from reviewbot.context import store_manager_ctx


@tool
def think(reasoning: str) -> str:
    """Record internal reasoning and thought process.

    Use this tool to think through problems, plan your approach, or reason about code before taking action.
    The reasoning is stored and will be included in subsequent requests to maintain context.

    Args:
        reasoning: Your internal thoughts, analysis, or reasoning about the current task.
                  This can include:
                  - Analysis of code patterns
                  - Planning next steps
                  - Reasoning about potential issues
                  - Conclusions drawn from observations

    Returns:
        Confirmation that the reasoning was recorded

    Examples:
        - "I notice this function has multiple responsibilities. It handles both data validation
           and API calls, which violates the Single Responsibility Principle."
        - "Before checking for issues, I should first understand the overall structure.
           The code appears to be a REST API with three main endpoints."
        - "This looks like a potential security issue - user input is being directly
           concatenated into a SQL query. I should flag this as high severity."
    """
    context = store_manager_ctx.get()
    issue_store = context.get("issue_store")

    if not issue_store:
        return "Context not available for storing reasoning."

    # Store reasoning in the issue store's metadata
    if not hasattr(issue_store, "_reasoning_history"):
        issue_store._reasoning_history = []

    issue_store._reasoning_history.append(reasoning)
    print("Reasoned:")
    print(reasoning)
    return f"Reasoning recorded: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}"
