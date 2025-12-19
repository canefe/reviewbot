import asyncio
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import dotenv
import requests
import typer
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr
from rich.console import Console

from reviewbot.agent.base import AgentRunnerInput, Settings, agent_runner
from reviewbot.context import store_manager_ctx
from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.infra.git.clone import clone_repo_persistent, get_repo_name
from reviewbot.models.gpt import get_gpt_model
from reviewbot.tools.compile_codebase import compile_codebase
from reviewbot.tools.search_codebase import (
    read_file,
    search_codebase,
    search_codebase_semantic_search,
)


@dataclass
class Config:
    llm_api_key: SecretStr
    llm_base_url: str
    llm_model_name: str
    gitlab_api_v4: str
    gitlab_token: str


def load_env() -> Config:
    dotenv.load_dotenv()
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_model_name = os.getenv("LLM_MODEL")
    gitlab_api_v4 = os.getenv("GITLAB_API_V4_URL")
    gitlab_token = os.getenv("GITLAB_BOT_TOKEN")
    if (
        not llm_api_key
        or not llm_base_url
        or not llm_model_name
        or not gitlab_api_v4
        or not gitlab_token
    ):
        raise ValueError(
            "LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, GITLAB_API_V4_URL, and GITLAB_BOT_TOKEN must be set"
        )
    return Config(
        llm_api_key=SecretStr(llm_api_key),
        llm_base_url=llm_base_url,
        llm_model_name=llm_model_name,
        gitlab_api_v4=gitlab_api_v4,
        gitlab_token=gitlab_token,
    )


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def test(
    repo_path: str = typer.Argument(..., help="Path to repository directory"),
):
    print(build_codebase_tree(Path(repo_path)))


def fetch_mr_diff(
    api_v4: str, project_id: str, mr_iid: str, token: str, timeout: int = 30
) -> str:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}

    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    diff_url = f"{mr_url}/raw_diffs"

    r = requests.get(mr_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    r = requests.get(diff_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    return r.text


def get_mr_branch(
    api_v4: str, project_id: str, mr_iid: str, token: str, timeout: int = 30
) -> str:
    api_v4 = api_v4.rstrip("/")
    headers = {"PRIVATE-TOKEN": token}
    mr_url = f"{api_v4}/projects/{project_id}/merge_requests/{mr_iid}"
    r = requests.get(mr_url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["source_branch"]


def tree(repo_path: Path) -> str:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()

    root = {}
    for file in out:
        parts = file.split("/")
        cur = root
        for p in parts:
            cur = cur.setdefault(p, {})

    def render(node, prefix=""):
        s = ""
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            s += prefix + ("└── " if last else "├── ") + name + "\n"
            if child:
                s += render(child, prefix + ("    " if last else "│   "))
        return s

    return render(root)


def build_codebase_tree(repo_dir: Path) -> str:
    return f"The codebase tree is:\n{tree(repo_dir)}"


def build_clone_url(api_v4: str, project_id: str, token: str) -> str:
    api_v4 = api_v4.rstrip("/")
    r = requests.get(
        f"{api_v4}/projects/{project_id}",
        headers={"PRIVATE-TOKEN": token},
    )
    r.raise_for_status()

    repo_url = r.json()["http_url_to_repo"]
    p = urlparse(repo_url)

    return urlunparse(
        (
            p.scheme,
            f"oauth2:{token}@{p.netloc}",
            p.path,
            "",
            "",
            "",
        )
    )


def build_user_prompt(repo_dir: Path, diff: str) -> str:
    return f"""
Review this GitLab merge request.

Hard rules:
- You are given tools to help you understand the codebase. You must use the tools to get greater context about the codebase. The code is already passed the compile check, do not assume there might be compiling issues.
- Use read_file tool to read the file at the given path, which can be retrieved from the search_codebase tool.
- Use the tool to get greater context about the codebase. You must understand the codebase before giving any valuable feedback.
- Every “issue” MUST include:
  - file path
  - a short quoted code fragment from the diff that proves it
  If you can’t quote evidence from the diff, DO NOT mention it.
- The code is already compiled, linted, and formatted, do not mention these issues.
- No tables.
Output format:
- Summary (1-3 bullets, diff-backed only)
- High-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Medium-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Low-risk issues (bullets; each bullet: file path + evidence quote + why it matters)
- Suggestions (bullets; include snippets when helpful)

Codebase tree:
{build_codebase_tree(repo_dir)}

Merge request diff:
{diff}
## High-risk issues (bullets, include file paths)
### Title_1
- Description_1
### Title_2
- Description_2
### Title_3
- Description_3

## Suggestions (bullets, include code snippets when helpful)
### Title_1
```diff
- var test := "test"
+ var test := "thats going to be a problem"
```
### Title_2
- Description_2
### Title_3
- Description_3

    """


def post_merge_request_note(
    api_v4: str,
    token: str,
    project_id: str,
    mr_iid: str,
    body: str,
    timeout: int = 30,
) -> None:
    url = f"{api_v4.rstrip('/')}/projects/{project_id}/merge_requests/{mr_iid}/notes"

    r = requests.post(
        url,
        headers={"PRIVATE-TOKEN": token},
        data={"body": body},
        timeout=timeout,
    )

    if r.status_code >= 300:
        raise RuntimeError(
            f"gitlab note post failed: {r.status_code} {r.reason}: {r.text}"
        )


def work_agent(api_v4: str, project_id: str, mr_iid: str, token: str):
    config = load_env()
    model = get_gpt_model(
        config.llm_model_name, config.llm_api_key, config.llm_base_url
    )

    clone_url = build_clone_url(api_v4, project_id, token)

    diff = fetch_mr_diff(api_v4, project_id, mr_iid, token)

    settings = Settings(max_tool_calls=10, max_iterations=10, max_retries=3)
    system_prompt = """
    You are a codebase reviewer. You are given a codebase tree and a merge request diff. You must review the codebase and the merge request diff and give a review of the codebase.
    """

    tools = [
        search_codebase,
        compile_codebase,
        read_file,
        search_codebase_semantic_search,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
    )
    branch = get_mr_branch(api_v4, project_id, mr_iid, token)
    repo_path = clone_repo_persistent(clone_url, branch=branch)
    repo_path = Path(repo_path).resolve()
    user_prompt = build_user_prompt(repo_path, diff)

    manager = CodebaseStoreManager()
    manager.set_repo_root(repo_path)
    manager.set_repo_name(get_repo_name(repo_path))
    store = manager.get_store()

    context = store_manager_ctx.set(manager)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    try:
        response = agent_runner.invoke(
            AgentRunnerInput(agent=agent, messages=messages, settings=settings)
        )

        if isinstance(response, str):
            # post_merge_request_note(api_v4, token, project_id, mr_iid, response)
            console.print(response)
            pass
        else:
            raise RuntimeError(f"Unexpected response type: {type(response)}")

    finally:
        store_manager_ctx.reset(context)


@app.command()
def work(
    project_id: str = typer.Argument(..., help="GitLab project ID"),
    mr_iid: str = typer.Argument(..., help="Merge request IID"),
):
    config = load_env()
    work_agent(
        config.gitlab_api_v4,
        project_id,
        mr_iid,
        config.gitlab_token,
    )


async def main():
    app()


if __name__ == "__main__":
    asyncio.run(main())
