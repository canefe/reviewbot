from importlib import resources

from langchain_core.prompts import ChatPromptTemplate

from reviewbot.core.config import Config


def get_prompt(prompt_name: str, config: Config) -> ChatPromptTemplate:
    """
    Load a prompt template from markdown file and return as ChatPromptTemplate.

    The markdown file should have this structure:
    ```
    # System Prompt

    [system message content]

    ---

    # Human Prompt

    [human message content]
    ```

    Args:
        prompt_name: Name of the prompt (without extension), can include subdirs like "review/quick_scan"
        config: Config object with optional custom_prompts_dir

    Returns:
        ChatPromptTemplate with system and human messages

    Raises:
        FileNotFoundError: If prompt file not found
    """
    filename = f"{prompt_name}.md"

    # 1. Try Custom Override
    prompt_content = None
    if config.custom_prompts_dir:
        custom_path = config.custom_prompts_dir / filename
        if custom_path.exists():
            prompt_content = custom_path.read_text(encoding="utf-8")

    # 2. Fallback to Package Default
    if prompt_content is None:
        try:
            prompt_content = (
                resources.files("reviewbot.agent.prompts")
                .joinpath(filename)
                .read_text(encoding="utf-8")
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Prompt '{filename}' not found in custom dir or package defaults."
            ) from e

    # Parse the markdown into system and human prompts
    system_prompt, human_prompt = _parse_prompt_markdown(prompt_content)

    # Create ChatPromptTemplate
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])  # pyright: ignore[reportUnknownMemberType]


def _parse_prompt_markdown(content: str) -> tuple[str, str]:
    """
    Parse a markdown prompt file into system and human message components.

    Expected format:
    # System Prompt

    [system content]

    ---

    # Human Prompt

    [human content]

    Args:
        content: Raw markdown content

    Returns:
        Tuple of (system_prompt, human_prompt)
    """
    # Split by the --- separator
    parts = content.split("---", 1)

    if len(parts) != 2:
        raise ValueError("Prompt markdown must have system and human sections separated by '---'")

    system_section = parts[0].strip()
    human_section = parts[1].strip()

    # Extract content after the first header (if present)
    system_prompt = _extract_prompt_content(system_section)
    human_prompt = _extract_prompt_content(human_section)

    return system_prompt, human_prompt


def _extract_prompt_content(section: str) -> str:
    """
        Extract the actual prompt content from a section, removing the header.
    r
        Args:
            section: A section of the markdown (system or human)

        Returns:
            The prompt content with header removed
    """
    lines = section.split("\n")

    # Skip the first line if it's a header (starts with #)
    start_idx = 0
    if lines and lines[0].strip().startswith("#"):
        start_idx = 1

    # Join remaining lines and strip
    content = "\n".join(lines[start_idx:]).strip()
    return content
