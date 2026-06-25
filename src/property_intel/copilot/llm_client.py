"""LLM client wrapper around the OpenAI chat completions API.

Supports both non-streaming (complete) and streaming (stream_complete) calls.
The raw OpenAI client is injected so tests can replace it with a mock without
touching the real API.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from property_intel.config import get_settings

Message = ChatCompletionMessageParam


class LLMClient:
    """Thin, injectable wrapper around OpenAI chat completions."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_settings(cls) -> LLMClient:
        settings = get_settings()
        return cls(
            client=OpenAI(api_key=settings.openai_api_key),
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
        )

    def complete(self, messages: list[Message]) -> str:
        """Return the full response as a single string."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def stream_complete(self, messages: list[Message]) -> Iterator[str]:
        """Yield text delta chunks as they arrive from the API."""
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
