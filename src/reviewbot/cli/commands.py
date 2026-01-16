from pathlib import Path

import typer

from reviewbot.agent.workflow import work_agent  # type: ignore
from reviewbot.cli.app import app
from reviewbot.infra.config.env import load_env
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.note import delete_discussion


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
    work_agent.invoke(  # type: ignore
        {
            "config": config,
            "project_id": project_id,
            "mr_iid": mr_iid,
        },
    )


@app.command()
def delete(
    project_id: str = typer.Argument(..., help="GitLab project ID"),
    mr_iid: str = typer.Argument(..., help="Merge request IID"),
    discussion_id: str = typer.Argument(..., help="Discussion ID"),
    note_id: str = typer.Argument(..., help="Note ID"),
):
    config = load_env()
    delete_discussion(
        config.gitlab_api_v4,
        config.gitlab_token,
        project_id,
        mr_iid,
        discussion_id,
        note_id,
    )
