from pathlib import Path

from langchain.tools import ToolRuntime, tool  # type: ignore
from rich.console import Console

from reviewbot.agent.workflow.state import CodebaseState

console = Console()


@tool
def ls_dir(
    dir_path: str,
    runtime: ToolRuntime,  # type: ignore
) -> str:
    """List the contents of a directory in the repository.

    Use this tool to explore directory structure and see what files and subdirectories exist.

    Args:
        dir_path: Relative path to the directory in the repository (e.g., "src" or "src/components")

    Returns:
        A formatted list of files and directories in the specified path

    Examples:
        - ls_dir("src") - List contents of src directory
        - ls_dir(".") - List contents of repository root
        - ls_dir("src/utils") - List contents of src/utils directory
    """
    if runtime.store is None:
        console.print("[red]ls_dir: Store not found in runtime[/red]")
        raise ValueError("Store not found in runtime")

    console.print(f"[cyan]ls_dir: '{dir_path}'[/cyan]")

    # Get codebase state from store
    NS = ("codebase",)
    raw = runtime.store.get(NS, "state")
    if not raw:
        console.print("[red]ls_dir: Codebase state not found in store[/red]")
        raise ValueError("Codebase state not found in store")

    codebase_data = raw.value if hasattr(raw, "value") else raw
    codebase = CodebaseState.model_validate(codebase_data)

    # Construct full path
    repo_root = Path(codebase.repo_root)
    normalized_path = Path(dir_path)
    full_path = repo_root / normalized_path

    console.print(f"[dim]  → Resolved path: {full_path}[/dim]")

    # Check if path exists
    if not full_path.exists():
        error_msg = (
            f"ERROR: Directory not found: '{dir_path}'. "
            f"Checked at: {full_path}. "
            "This directory may not exist or the path may be incorrect."
        )
        console.print("[red]  → Returning: DIRECTORY NOT FOUND error[/red]")
        return error_msg

    if not full_path.is_dir():
        error_msg = (
            f"ERROR: Path is not a directory: '{dir_path}'. Use read_file() to read files instead."
        )
        console.print("[red]  → Returning: NOT A DIRECTORY error[/red]")
        return error_msg

    # List directory contents
    console.print("[green]  ✓ Directory exists, listing contents...[/green]")
    try:
        entries = sorted(full_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))

        if not entries:
            console.print("[dim]  → Directory is empty[/dim]")
            return f"Directory '{dir_path}' is empty."

        # Format output
        output_lines = [f"Contents of '{dir_path}':\n"]
        dirs: list[str] = []
        files: list[str] = []

        for entry in entries:
            relative_name = entry.name
            if entry.is_dir():
                dirs.append(f"  [DIR]  {relative_name}/")
            else:
                # Get file size
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                files.append(f"  [FILE] {relative_name:<40} {size_str:>10}")

        # Add directories first, then files
        output_lines.extend(dirs)
        output_lines.extend(files)

        result = "\n".join(output_lines)
        console.print(f"[green]  → Found {len(dirs)} directories and {len(files)} files[/green]")
        console.print(f"[dim]  → Preview: {result[:100]}...[/dim]")
        return result

    except PermissionError:
        error_msg = f"ERROR: Permission denied accessing directory: {dir_path}"
        console.print("[red]  → Returning: PERMISSION ERROR[/red]")
        return error_msg
    except Exception as e:
        error_msg = f"ERROR: Failed to list directory '{dir_path}': {str(e)}"
        console.print(f"[red]  → Returning: UNEXPECTED ERROR: {e}[/red]")
        return error_msg
