from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task  # type:ignore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from reviewbot.infra.git.clone import clone_repo_persistent
from reviewbot.infra.git.repo_tree import tree
from reviewbot.infra.gitlab.clone import build_clone_url
from reviewbot.infra.gitlab.diff import FileDiff, fetch_mr_diffs, get_mr_branch

in_memory_checkpointer = InMemorySaver()
in_memory_store = InMemoryStore()


class GitLabData(BaseModel):
    clone_url: str
    diffs: list[FileDiff]
    diff_refs: dict[str, str]
    branch: str


class RepoSnapshot(BaseModel):
    repo_path: str
    repo_tree: str


@task
def fetch_gitlab_data(
    api_v4: str,
    project_id: str,
    mr_iid: str,
    token: str,
) -> GitLabData:
    clone_url = build_clone_url(api_v4, project_id, token)
    diffs, diff_refs = fetch_mr_diffs(api_v4, project_id, mr_iid, token)
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    return GitLabData(
        clone_url=clone_url,
        diffs=diffs,
        diff_refs=diff_refs,
        branch=branch,
    )


@task
def clone_and_tree(clone_url: str, branch: str) -> RepoSnapshot:
    repo_path = Path(clone_repo_persistent(clone_url, branch=branch)).resolve()
    return RepoSnapshot(
        repo_path=str(repo_path),
        repo_tree=tree(repo_path),
    )
