import json
from dataclasses import dataclass, field
from typing import Any, List

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.func import entrypoint, task
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console

console = Console()


async def generate_response(
    agent: CompiledStateGraph[dict, None, None, None], messages: List[AnyMessage]
) -> str:
    flavor_prompt = """
    You must generate a final response now. Do not call any tools any longer. Use what you got.
    """
    flavor_message = SystemMessage(content=flavor_prompt)
    messages.append(flavor_message)
    finished = False
    print(f"Generating response with {len(messages)} messages")
    while not finished:
        try:
            response = await agent.ainvoke({"messages": messages})
            if isinstance(response, dict):
                response = response.get("content", response)
            if isinstance(response, list):
                response = response[-1].get("text", response[-1])
            if isinstance(response, dict):
                response = response.get("messages", response)
                if isinstance(response, list):
                    response = response[-1]
                    if isinstance(response, dict):
                        response = response.get("text", response)
                    else:
                        response = response.text
            if response != None and response != "":
                finished = True
        except Exception:
            # print traceback
            import traceback

            traceback.print_exc()
    return response


@dataclass
class Settings:
    """Settings for the agent runner"""

    max_tool_calls: int = -1
    """Maximum number of tool calls
    -1 for unlimited
    """
    max_iterations: int = -1
    """Maximum number of iterations
    -1 for unlimited
    """
    max_retries: int = -1
    """Maximum number of retries
    -1 for unlimited
    """


# Generate response workflow model
@dataclass
class AgentRunnerInput:
    agent: Any
    messages: List[AnyMessage]
    settings: Settings = field(default_factory=Settings)


@task
def analyzer(agent: Any, initial_prompt: str, response: str) -> str:
    """Analyze the response and rewrite it if necessary to be more helpful."""
    system_prompt = """
    You are a response analyzer. The response is a review of a codebase. You must analyze the response and rewrite it if necessary to be more helpful. For example, the review might contain assumptions or unverified issues. You can use the tools provided 'search_codebase' and 'read_file' to get more context about the codebase. Verify whether these issues are actually exists and not just assumptions.

    You should also during your analysis, find out actual problems in the codebase and suggest solutions to them like the provided response.
    """
    human_prompt = f"""
    Initial prompt: {initial_prompt}
    Generated review for above prompt: {response}
    Analyze the review and rewrite it if necessary to be more helpful.
    Your review should be scoped to the merge request diff only and its related codebase changes.
    You are not allowed to use tables. Use headings and lists to structure your response.
    - Use read_file tool to read the file at the given path, which can be retrieved from the search_codebase tool.
    - Use the tool to get greater context about the codebase. You must understand the codebase before giving any valuable feedback.
    - Every “issue” MUST include:
    - file path
    - a short quoted code fragment from the diff that proves it
    If you can’t quote evidence from the diff, DO NOT mention it.
    - The code is already compiled, linted, and formatted. It is impossible for the code to contain any missing methods, or using non-existent methods.
    - No tables.
    Output format:
    - Summary (1 small paragraph, diff-backed only, also commending the attempt of the merge requester's code changes.)
    - High-risk issues (bullets; each bullet: file path + evidence quote + why it matters; only use this section if its CRITICAL.)
    - Medium-risk issues (bullets; each bullet: file path + evidence quote + why it matters; only use this section if its MEDIUM.)
    - Low-risk issues (bullets; each bullet: file path + evidence quote + why it matters; only use this section if its LOW.)
    - Suggestions (bullets; include snippets when helpful)
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
    response = tool_caller(agent, messages, Settings(max_tool_calls=20)).result()
    console.print(response)
    return response


@task
def tool_caller(agent: Any, messages: List[AnyMessage], settings: Settings) -> str:
    finished = False
    final_response = None
    max_tool_calls = settings.max_tool_calls

    while not finished:
        try:
            for chunk in agent.stream({"messages": messages}, stream_mode="values"):
                latest_message = chunk["messages"][-1]

                if isinstance(latest_message, AIMessage):
                    content = latest_message.content
                    if isinstance(content, list):
                        if content and "content" in content[-1]:
                            reason = content[-1]["content"][-1]["text"]
                        elif content and "text" in content[-1]:
                            reason = content[-1]["text"]
                        else:
                            reason = str(content)
                    else:
                        reason = content
                    print(f"AI response: {reason}")
                    final_response = content
                    messages.append(latest_message)

                elif isinstance(latest_message, ToolMessage):
                    print("Called a tool!", f"{max_tool_calls} left")
                    if max_tool_calls != -1:
                        max_tool_calls -= 1
                        if max_tool_calls <= 0:
                            finished = True
                            break

            if final_response:
                if isinstance(final_response, list):
                    last = final_response[-1]
                    if last.get("content"):
                        final_response = last["content"][-1]["text"]
                    elif last.get("text"):
                        final_response = last["text"]
                    else:
                        final_response = last

                if isinstance(final_response, dict):
                    print(f"Final response: {final_response}")
                    if final_response.get("content"):
                        final_response = final_response["content"][-1]["text"]
                    elif final_response.get("text"):
                        final_response = final_response["text"]
                    else:
                        final_response = final_response

                if not (
                    isinstance(final_response, str)
                    and final_response.startswith("{")
                    and final_response.endswith("}")
                ):
                    finished = True
                else:
                    final_response = json.loads(final_response.replace("\n", ""))
                    print(f"Bugged AI response: {final_response}")

        except Exception:
            import traceback

            traceback.print_exc()

    if not isinstance(final_response, str):
        console.print("Final response is not a string, returning None")
        return "None"
    return final_response


@entrypoint()
def agent_runner(input: AgentRunnerInput):
    agent = input.agent
    messages = input.messages
    settings = input.settings

    final_response = tool_caller(agent, messages, settings).result()

    console.print("Analyzing response...")
    print(messages)
    final_response = analyzer(agent, messages[-1].content, final_response).result()
    console.print(f"Final response: {final_response}")
    return final_response


@entrypoint()
async def agent_runner_async(input: AgentRunnerInput):
    """
    A base workflow to be used with every agent.
    """
    agent = input.agent
    messages = input.messages
    settings = input.settings

    finished = False
    final_response = None
    max_tool_calls = settings.max_tool_calls

    while not finished:
        try:
            async for chunk in agent.astream(
                {"messages": messages}, stream_mode="values"
            ):
                # Each chunk contains the full state at that point
                latest_message = chunk["messages"][-1]
                if isinstance(latest_message, AIMessage):
                    reason = latest_message.content[0].get("content", [])
                    if len(reason) > 0:
                        reason = reason[-1]["text"]
                    else:
                        reason = latest_message.content
                    print(reason)
                    final_response = latest_message.content
                    messages.append(latest_message)
                elif isinstance(latest_message, ToolMessage):
                    print("Called a tool!", f"{max_tool_calls} left")
                    if max_tool_calls != -1:
                        max_tool_calls -= 1
                        if max_tool_calls <= 0:
                            finished = True
            if final_response:
                if isinstance(final_response, list):
                    if final_response[-1].get("content"):
                        final_response = final_response[-1]["content"][-1]["text"]
                    elif final_response[-1].get("text"):
                        final_response = final_response[-1]["text"]
                    else:
                        final_response = final_response[-1]
                # check if final response is a bugged AI message (last message["content"][-1]["text"] starts with { and ends with })
                if not final_response.startswith("{") and not final_response.endswith(
                    "}"
                ):
                    finished = True
                else:
                    # Parse the final response as JSON (might contain \n s)
                    final_response = json.loads(final_response.replace("\n", ""))
                    print(f"Bugged AI response: {final_response}")
                    if finished:
                        final_response = await generate_response(agent, messages)
        except Exception:
            # print traceback
            import traceback

            traceback.print_exc()

    return final_response
