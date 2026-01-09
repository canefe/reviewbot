from abc import ABC, abstractmethod

from pydantic import BaseModel, SecretStr


class GitProviderConfig(BaseModel, ABC):
    """Abstract base configuration for git providers."""

    token: SecretStr

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }

    @abstractmethod
    def get_api_base_url(self) -> str:
        """Return the base API URL for this provider."""
        pass

    @abstractmethod
    def get_project_identifier(self) -> str:
        """Return the project/repository identifier."""
        pass

    @abstractmethod
    def get_pr_identifier(self) -> str:
        """Return the pull/merge request identifier."""
        pass


class GitLabConfig(GitProviderConfig):
    """GitLab-specific configuration."""

    api_v4: str
    project_id: str
    mr_iid: str

    def get_api_base_url(self) -> str:
        return self.api_v4

    def get_project_identifier(self) -> str:
        return self.project_id

    def get_pr_identifier(self) -> str:
        return self.mr_iid
