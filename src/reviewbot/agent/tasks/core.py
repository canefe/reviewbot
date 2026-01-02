import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from rich.console import Console

console = Console()


@dataclass
class ToolCallerSettings:
    """Tool caller settings"""

    max_tool_calls: int = -1
    """Maximum number of tool calls
    -1 for unlimited
    """
    max_iterations: int = -1
    """Maximum number of iterations
    -1 for unlimited
    """
    max_retries: int = 3
    """Maximum number of retries for failed API calls
    Default: 3 attempts
    """
    retry_delay: float = 1.0
    """Initial retry delay in seconds
    Will use exponential backoff: delay * (2 ** attempt)
    Default: 1.0 second
    """
    retry_max_delay: float = 60.0
    """Maximum retry delay in seconds
    Default: 60 seconds
    """


def tool_caller(agent: Any, messages: list[BaseMessage], settings: ToolCallerSettings) -> str:
    finished = False
    final_response = None
    max_tool_calls = settings.max_tool_calls
    total_tool_calls = 0

    while not finished:
        try:
            # Invoke the agent with current messages
            result = agent.invoke({"messages": messages})

            # Get the latest message from result
            latest_message = result["messages"][-1]

            # Update messages for next iteration
            messages = result["messages"]

            if isinstance(latest_message, AIMessage):
                # Check if this message has tool calls
                if latest_message.tool_calls:
                    # Agent wants to use tools, continue loop
                    console.print(
                        f"[dim]Agent is using {len(latest_message.tool_calls)} tools[/dim]"
                    )
                    continue
                else:
                    # No tool calls = final response
                    content = latest_message.content
                    if isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif "text" in block:
                                    text_parts.append(block["text"])
                        final_response = "\n".join(text_parts) if text_parts else str(content)
                    else:
                        final_response = content

                    console.print(f"[dim]Got final response: {final_response[:100]}...[/dim]")
                    finished = True

            elif isinstance(latest_message, ToolMessage):
                total_tool_calls += 1
                console.print(f"[dim]Tool call completed ({total_tool_calls} total)[/dim]")
                if max_tool_calls != -1 and total_tool_calls >= max_tool_calls:
                    console.print(
                        f"[yellow]Max tool calls ({max_tool_calls}) reached - forcing final response[/yellow]"
                    )
                    # Force the agent to provide a final response
                    messages.append(
                        HumanMessage(
                            content="You have reached the maximum number of tool calls. Please provide your final response now in the required JSON format. If you haven't found any issues, return an empty array: []"
                        )
                    )
                    # Get one final response from the agent
                    try:
                        result = agent.invoke({"messages": messages})
                        latest_message = result["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            final_response = latest_message.content
                            if isinstance(final_response, list):
                                text_parts = []
                                for block in final_response:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text_parts.append(block.get("text", ""))
                                final_response = (
                                    "\n".join(text_parts) if text_parts else str(final_response)
                                )
                        else:
                            final_response = "[]"  # Empty array as fallback
                    except Exception as e:
                        console.print(f"[red]Error getting forced response: {e}[/red]")
                        final_response = "[]"  # Empty array as fallback
                    finished = True

        except Exception as e:
            import traceback

            console.print(f"[red]Error in tool_caller: {e}[/red]")
            traceback.print_exc()
            finished = True
            final_response = None

    if not isinstance(final_response, str):
        console.print(f"Final response is not a string: {final_response}, returning None")
        return "None"
    return final_response


def tool_caller_stream(
    agent: Any, messages: list[BaseMessage], settings: ToolCallerSettings
) -> str:
    finished = False
    final_response = None
    max_tool_calls = settings.max_tool_calls
    last_chunk = None
    tool_call_count = 0

    while not finished:
        try:
            for chunk in agent.stream({"messages": messages}, stream_mode="values"):
                last_chunk = chunk
                latest_message = chunk["messages"][-1]

                if isinstance(latest_message, AIMessage):
                    final_response = latest_message.content

                elif isinstance(latest_message, ToolMessage):
                    tool_call_count += 1
                    print(f"Called a tool! {tool_call_count} calls made")
                    if max_tool_calls != -1 and tool_call_count >= max_tool_calls:
                        finished = True
                        break

            if last_chunk:
                messages = last_chunk["messages"]
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
                    if final_response.get("content"):
                        final_response = final_response["content"][-1]["text"]
                    elif final_response.get("text"):
                        final_response = final_response["text"]
                    else:
                        console.print("Messages:")
                        # get last 5 messages
                        console.print(messages[:5])
                        console.print("Popping message:")
                        console.print(messages[-1])
                        console.print(messages.pop())
                        finished = False
                        continue

                if isinstance(final_response, str):
                    try:
                        final_response = json.loads(final_response)
                        # valid JSON → keep looping
                    except json.JSONDecodeError:
                        finished = True
                else:
                    final_response = json.loads(final_response.replace("\n", ""))

        except Exception:
            import traceback

            traceback.print_exc()

            if last_chunk:
                console.print("Chunk:")
                console.print(last_chunk)
                messages = last_chunk["messages"]
    if not isinstance(final_response, str):
        console.print(f"Final response is not a string : {final_response}, returning None")
        return "None"
    return final_response
