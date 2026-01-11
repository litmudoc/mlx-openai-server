"""Tests for Solar Open reasoning and tool call parsers."""

from __future__ import annotations

from app.handler.parser.solar_open import SolarOpenThinkingParser, SolarOpenToolParser


def test_parse_reasoning_and_content() -> None:
    """Ensure reasoning/content parsing works for completed responses."""
    parser = SolarOpenThinkingParser()
    reasoning, content = parser.parse("<|think|>Reason<|end|><|content|>Hello")

    assert reasoning == "Reason"
    assert content == "Hello"


def test_parse_ignores_begin_assistant_tag() -> None:
    """Ensure the leading assistant header tag is stripped."""
    parser = SolarOpenThinkingParser()
    reasoning, content = parser.parse("<|begin|>assistant<|content|>Hi")

    assert reasoning is None
    assert content == "Hi"


def test_parse_tool_calls() -> None:
    """Ensure tool call parsing extracts name and arguments."""
    parser = SolarOpenToolParser()
    text = (
        "<|tool_calls|><|tool_call:begin|>id"
        "<|tool_call:name|>weather"
        "<|tool_call:args|>{\"city\": \"Seoul\"}"
        "<|tool_call:end|><|calls|>"
    )
    tool_calls, content = parser.parse(text)

    assert content == ""
    assert tool_calls == [{"name": "weather", "arguments": {"city": "Seoul"}}]


def test_stream_reasoning_to_content() -> None:
    """Ensure streaming reasoning transitions to content properly."""
    parser = SolarOpenThinkingParser()

    res1, complete1 = parser.parse_stream("<|think|>Hello ")
    assert complete1 is False
    assert res1 == {"reasoning_content": "Hello "}

    res2, complete2 = parser.parse_stream("world<|end|><|content|>Hi")
    assert complete2 is True
    assert res2 == {"reasoning_content": "world", "content": "Hi"}


def test_stream_tool_call() -> None:
    """Ensure tool call streaming emits name then arguments."""
    parser = SolarOpenToolParser()

    res1, complete1 = parser.parse_stream("Hi <|tool_calls|>")
    assert complete1 is True
    assert res1 == {"content": "Hi "}

    res2, complete2 = parser.parse_stream(
        "<|tool_call:begin|>id<|tool_call:name|>weather"
        "<|tool_call:args|>{\"city\":"
    )
    assert complete2 is True
    assert res2 == {"name": "weather"}

    res3, complete3 = parser.parse_stream("\"Seoul\"}<|tool_call:end|>")
    assert complete3 is True
    assert res3 == {"arguments": "{\"city\":\"Seoul\"}"}
