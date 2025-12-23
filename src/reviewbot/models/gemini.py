from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


def get_gemini_model(model_name: str, api_key: SecretStr, project_id: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        project=project_id,
        thinking_level="medium",
    )
