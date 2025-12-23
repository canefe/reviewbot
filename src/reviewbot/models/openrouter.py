import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, api_key: Optional[SecretStr] = None, **kwargs: Any):
        api_key = api_key or SecretStr(os.getenv("OPENROUTER_API_KEY", ""))
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            **kwargs,
        )


def get_openrouter_model(model_name: str, api_key: SecretStr):
    return ChatOpenRouter(
        model_name=model_name,
    )
