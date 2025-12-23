from pathlib import Path

import typer

from reviewbot.agent.workflow import work_agent
from reviewbot.cli.app import app
from reviewbot.infra.config.env import load_env
from reviewbot.infra.git.repo_tree import tree


@app.command()
def test(
    repo_path: str = typer.Argument(..., help="Path to repository directory"),
):
    print(tree(Path(repo_path)))


@app.command()
def work(
    project_id: str = typer.Argument(..., help="GitLab project ID"),
    mr_iid: str = typer.Argument(..., help="Merge request IID"),
):
    config = load_env()
    work_agent(
        config,
        project_id,
        mr_iid,
    )
