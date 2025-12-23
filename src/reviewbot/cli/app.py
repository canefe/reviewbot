import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
console = Console()

# side-effect import: registers commands
from reviewbot.cli import (  # noqa: F401, E402
    commands,  # pyright: ignore
)
