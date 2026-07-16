from typing import Tuple
import os
from openai import AsyncOpenAI


async def get_openai_client() -> Tuple[AsyncOpenAI, str]:
    """
    Instantiates the OpenAI client asynchronously and returns the client and default model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")

    _client = AsyncOpenAI(api_key=api_key)
    return _client, model