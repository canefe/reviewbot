import os

import dotenv
from pydantic import SecretStr

from reviewbot.core.config import Config


def load_env() -> Config:
    dotenv.load_dotenv()
    llm_api_key = os.getenv("GOOGLE_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_model_name = os.getenv("LLM_MODEL")
    gitlab_api_v4 = os.getenv("GITLAB_API_V4_URL")
    gitlab_token = os.getenv("GITLAB_BOT_TOKEN")
    gemini_project_id = os.getenv("GEMINI_PROJECT_ID")
    if (
        not llm_api_key
        or not llm_base_url
        or not llm_model_name
        or not gitlab_api_v4
        or not gitlab_token
        or not gemini_project_id
    ):
        raise ValueError(
            "LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, GITLAB_API_V4_URL, GITLAB_BOT_TOKEN, and GEMINI_PROJECT_ID must be set"
        )
    return Config(
        llm_api_key=SecretStr(llm_api_key),
        llm_base_url=llm_base_url,
        llm_model_name=llm_model_name,
        gitlab_api_v4=gitlab_api_v4,
        gitlab_token=gitlab_token,
        gemini_project_id=gemini_project_id,
    )
