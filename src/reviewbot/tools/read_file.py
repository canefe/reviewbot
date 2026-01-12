from pathlib import Path

from langchain.tools import ToolRuntime, tool  # type: ignore
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState

console = Console()


@tool
def read_file(
    file_path: str,
    runtime: ToolRuntime,  # type: ignore
    line_start: int | None = None,
    line_end: int | None = None,
) -> str:
    """Read the contents of a file from the repository.

    Use this tool to get the full context of a file when the diff alone is not sufficient
    to understand the code. This helps avoid false positives by seeing the complete picture.

    Args:
        file_path: Relative path to the file in the repository (e.g., "src/main.go")
        line_start: Optional line number to start reading from (1-indexed)
        line_end: Optional line number to stop reading at (inclusive)

    Returns:
        The file contents, optionally limited to the specified line range

    Examples:
        - read_file("src/main.go") - Read entire file
        - read_file("src/main.go", line_start=10, line_end=50) - Read lines 10-50

    Note:
        Returns an error message if the file is newly added (doesn't exist in current checkout)
    """
    if runtime.store is None:
        console.print("[red]read_file: Store not found in runtime[/red]")
        raise ValueError("Store not found in runtime")

    line_range = f" (lines {line_start}-{line_end})" if line_start or line_end else ""
    console.print(f"[cyan]read_file: '{file_path}'{line_range}[/cyan]")

    # Get codebase state from store
    NS = ("codebase",)
    raw = runtime.store.get(NS, "state")
    if not raw:
        console.print("[red]read_file: Codebase state not found in store[/red]")
        raise ValueError("Codebase state not found in store")

    codebase_data = raw.value if hasattr(raw, "value") else raw
    codebase = CodebaseState.model_validate(codebase_data)

    # Construct full path, handling different path separators
    repo_root = Path(codebase.repo_root)
    # Normalize the file path (convert to Path and resolve)
    normalized_path = Path(file_path)
    full_path = repo_root / normalized_path

    console.print(f"[dim]  → Resolved path: {full_path}[/dim]")

    # Check if file exists
    if not full_path.exists():
        console.print(f"[yellow]  File does not exist at: {full_path}[/yellow]")

        # Check if this is a new file (added in the diff)
        diffs = codebase.diffs
        is_new_file = any(
            d.new_path == file_path and (d.is_new_file or not d.old_path) for d in diffs
        )

        if is_new_file:
            error_msg = (
                f"ERROR: Cannot read '{file_path}' - This is a NEW FILE being added in this change. "
                "The file doesn't exist in the current checkout, only in the diff. "
                "You can only see the changes in the diff via get_diff(). "
                "Since this is a new file, you cannot verify imports or variables from elsewhere in the file - "
                "assume the developer has the complete context and do not flag missing imports/variables as issues."
            )
            console.print("[yellow]  → Returning: NEW FILE error[/yellow]")
            console.print(f"[dim]  → Message: {error_msg[:100]}...[/dim]")
            return error_msg

        # File truly doesn't exist
        error_msg = (
            f"ERROR: File not found: '{file_path}'. "
            f"Checked at: {full_path}. "
            "This file may have been deleted, renamed, or the path may be incorrect. "
            "Available files in this diff can be checked with get_diff() tool."
        )
        console.print("[red]  → Returning: FILE NOT FOUND error[/red]")
        console.print(f"[dim]  → Message: {error_msg[:100]}...[/dim]")
        return error_msg

    if not full_path.is_file():
        if full_path.is_dir():
            error_msg = (
                f"ERROR: Path is a directory, not a file: '{file_path}'. "
                f"Use ls_dir('{file_path}') to list the contents of this directory instead."
            )
        else:
            error_msg = f"ERROR: Path is not a file: {file_path}"
        console.print("[red]  → Returning: NOT A FILE error[/red]")
        console.print(f"[dim]  → Message: {error_msg}[/dim]")
        return error_msg

    # Read file
    console.print("[green]  ✓ File exists, reading...[/green]")
    try:
        with open(full_path, encoding="utf-8") as f:
            if line_start is None and line_end is None:
                content = f.read()
                num_lines = content.count("\n") + 1
                num_chars = len(content)
                console.print(
                    f"[green]  → Successfully read entire file: {num_lines} lines, {num_chars} chars[/green]"
                )
                console.print(f"[dim]  → Preview: {content[:100]}...[/dim]")
                return content

            lines = f.readlines()
            total_lines = len(lines)

            # Adjust indices (convert to 0-based)
            start_idx = (line_start - 1) if line_start else 0
            end_idx = line_end if line_end else total_lines

            # Auto-adjust range to file bounds (be lenient)
            if start_idx < 0:
                console.print(f"[yellow]  Adjusting line_start from {line_start} to 1[/yellow]")
                start_idx = 0
            if start_idx >= total_lines:
                console.print(
                    f"[yellow]  line_start {line_start} exceeds file length ({total_lines}), using last line[/yellow]"
                )
                start_idx = max(0, total_lines - 1)

            if end_idx > total_lines:
                console.print(
                    f"[yellow]  Adjusting line_end from {line_end} to {total_lines} (file length)[/yellow]"
                )
                end_idx = total_lines

            if start_idx >= end_idx:
                console.print(
                    f"[yellow]  Invalid range ({line_start}-{line_end}), reading entire file instead[/yellow]"
                )
                content = "".join(lines)
                num_lines = len(lines)
                num_chars = len(content)
                console.print(
                    f"[green]  → Successfully read entire file: {num_lines} lines, {num_chars} chars[/green]"
                )
                return content

            # Return selected lines with line numbers
            result_lines: list[str] = []
            for i in range(start_idx, end_idx):
                line_num = i + 1
                result_lines.append(f"{line_num:4d} | {lines[i]}")

            content = "".join(result_lines)
            num_lines_returned = end_idx - start_idx
            console.print(
                f"[green]  → Successfully read lines {line_start}-{line_end}: {num_lines_returned} lines[/green]"
            )
            console.print(f"[dim]  → Preview: {content[:100]}...[/dim]")
            return content

    except UnicodeDecodeError as e:
        error_msg = f"File is not a text file or uses unsupported encoding: {file_path}"
        console.print("[red]  → Returning: ENCODING ERROR[/red]")
        console.print(f"[dim]  → Message: {error_msg}[/dim]")

        raise ValueError(error_msg) from e
