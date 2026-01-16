from pathlib import Path

from pydantic import BaseModel, SecretStr


class Config(BaseModel):
    llm_api_key: SecretStr
    llm_base_url: str
    llm_model_name: str
    gitlab_api_v4: str
    gitlab_token: str
    gemini_project_id: str
    create_threads: bool = False
    custom_prompts_dir: Path | None = None

    model_config = {
        "extra": "forbid",
        "frozen": True,
    }
