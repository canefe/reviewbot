from urllib.parse import urlparse, urlunparse

import requests


def build_clone_url(api_v4: str, project_id: str, token: str) -> str:
    api_v4 = api_v4.rstrip("/")
    r = requests.get(
        f"{api_v4}/projects/{project_id}",
        headers={"PRIVATE-TOKEN": token},
    )
    r.raise_for_status()

    repo_url = r.json()["http_url_to_repo"]
    p = urlparse(repo_url)  # type: ignore

    return urlunparse(
        (
            p.scheme,  # type: ignore
            f"oauth2:{token}@{p.netloc}",  # type: ignore
            p.path,  # type: ignore
            "",
            "",
            "",
        )  # type: ignore
    )
