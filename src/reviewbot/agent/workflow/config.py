from dataclasses import dataclass


@dataclass
class GitLabConfig:
    """GitLab API configuration"""

    api_v4: str
    token: str
    project_id: str
    mr_iid: str
