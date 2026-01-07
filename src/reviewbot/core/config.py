from dataclasses import dataclass

from pydantic import SecretStr


@dataclass
class Config:
    llm_api_key: SecretStr
    llm_base_url: str
    llm_model_name: str
    gitlab_api_v4: str
    gitlab_token: str
    gemini_project_id: str
    create_threads: bool = False
