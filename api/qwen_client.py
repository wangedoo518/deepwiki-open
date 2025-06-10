"""Qwen ModelClient integration via DashScope compatible API."""
import os

from typing import Optional
from .openai_client import OpenAIClient


class QwenClient(OpenAIClient):
    """A thin wrapper around :class:`OpenAIClient` for Alibaba Qwen service.

    It uses DashScope's OpenAI-compatible endpoint by default so it can share the
    same logic as ``OpenAIClient``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_base_url_name: str = "DASHSCOPE_BASE_URL",
        env_api_key_name: str = "DASHSCOPE_API_KEY",
        *args,
        **kwargs,
    ) -> None:
        base_url = base_url or os.getenv(
            env_base_url_name, "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            env_base_url_name=env_base_url_name,
            env_api_key_name=env_api_key_name,
            *args,
            **kwargs,
        )
