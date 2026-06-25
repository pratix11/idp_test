"""Tests for Task 26: LLM client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from property_intel.copilot.llm_client import LLMClient, Message


def _make_client(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[LLMClient, MagicMock]:
    mock_openai = MagicMock()
    return LLMClient(client=mock_openai, model=model, temperature=temperature, max_tokens=max_tokens), mock_openai


# ── complete() ─────────────────────────────────────────────────────────────────

def test_complete_returns_content() -> None:
    client, mock_openai = _make_client()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello world"
    mock_openai.chat.completions.create.return_value = mock_response

    result = client.complete([{"role": "user", "content": "Hi"}])

    assert result == "Hello world"


def test_complete_passes_correct_params() -> None:
    client, mock_openai = _make_client(model="gpt-4o", temperature=0.7, max_tokens=256)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_openai.chat.completions.create.return_value = mock_response

    messages: list[Message] = [{"role": "user", "content": "test"}]
    client.complete(messages)

    mock_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        stream=False,
    )


def test_complete_returns_empty_string_when_content_is_none() -> None:
    client, mock_openai = _make_client()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_openai.chat.completions.create.return_value = mock_response

    result = client.complete([{"role": "user", "content": "hi"}])

    assert result == ""


def test_complete_passes_system_and_user_messages() -> None:
    client, mock_openai = _make_client()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "answer"
    mock_openai.chat.completions.create.return_value = mock_response

    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"},
    ]
    client.complete(messages)

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == messages


# ── stream_complete() ──────────────────────────────────────────────────────────

def test_stream_complete_yields_non_none_chunks() -> None:
    client, mock_openai = _make_client()

    chunk1, chunk2, chunk3 = MagicMock(), MagicMock(), MagicMock()
    chunk1.choices[0].delta.content = "Hello"
    chunk2.choices[0].delta.content = " world"
    chunk3.choices[0].delta.content = None  # should be skipped
    mock_openai.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    result = list(client.stream_complete([{"role": "user", "content": "hi"}]))

    assert result == ["Hello", " world"]


def test_stream_complete_uses_stream_true() -> None:
    client, mock_openai = _make_client()
    mock_openai.chat.completions.create.return_value = iter([])

    list(client.stream_complete([{"role": "user", "content": "hi"}]))

    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True


def test_stream_complete_empty_stream_yields_nothing() -> None:
    client, mock_openai = _make_client()
    mock_openai.chat.completions.create.return_value = iter([])

    result = list(client.stream_complete([{"role": "user", "content": "hi"}]))

    assert result == []


def test_stream_complete_all_none_deltas_yields_nothing() -> None:
    client, mock_openai = _make_client()

    chunk = MagicMock()
    chunk.choices[0].delta.content = None
    mock_openai.chat.completions.create.return_value = iter([chunk])

    result = list(client.stream_complete([{"role": "user", "content": "test"}]))

    assert result == []


# ── from_settings() ────────────────────────────────────────────────────────────

def test_from_settings_uses_config_values() -> None:
    mock_settings = MagicMock()
    mock_settings.openai_api_key = "sk-test-key"
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_temperature = 0.3
    mock_settings.openai_max_tokens = 2048

    with patch("property_intel.copilot.llm_client.get_settings", return_value=mock_settings):
        with patch("property_intel.copilot.llm_client.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            c = LLMClient.from_settings()

    mock_openai_cls.assert_called_once_with(api_key="sk-test-key")
    assert c.model == "gpt-4o"
    assert c.temperature == 0.3
    assert c.max_tokens == 2048
