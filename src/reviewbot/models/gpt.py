import os

import dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

dotenv.load_dotenv()

llm_api_key = os.getenv("LLM_API_KEY")
llm_base_url = os.getenv("LLM_BASE_URL")
llm_model_name = os.getenv("LLM_MODEL")


def get_gpt_model(
    llm_model_name: str,
    llm_api_key: SecretStr,
    base_url: str,
    temperature: float = 0.2,
    reasoning_effort: str = "low",
):
    return ChatOpenAI(
        model=llm_model_name,
        api_key=llm_api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


def get_gpt_model_low_effort(
    llm_model_name: str,
    llm_api_key: SecretStr,
    base_url: str,
    temperature: float = 0.2,
):
    """Get a GPT model with low reasoning effort for simple tasks like greetings."""
    return ChatOpenAI(
        model=llm_model_name,
        api_key=llm_api_key,
        base_url=base_url,
        temperature=temperature,
        reasoning_effort="low",
    )
