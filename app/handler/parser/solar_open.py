"""Solar Open parser implementations for reasoning and tool call streams."""

from __future__ import annotations

import ast
import json
from typing import Any

THINK_TAG = "<|think|>"
END_TAG = "<|end|>"
BEGIN_ASSISTANT_TAG = "<|begin|>assistant"
CONTENT_TAG = "<|content|>"
TOOL_CALLS_TAG = "<|tool_calls|>"
TOOL_CALL_BEGIN = "<|tool_call:begin|>"
TOOL_CALL_NAME = "<|tool_call:name|>"
TOOL_CALL_ARGS = "<|tool_call:args|>"
TOOL_CALL_END = "<|tool_call:end|>"
CALLS_END = "<|calls|>"
FLUSH_TAG = "<|flush|>"


class SolarOpenThinkingParser:
    """Parser for Solar Open reasoning tags and content boundaries."""

    def __init__(self) -> None:
        """Initialize the parser with streaming state."""
        self._buffer = ""
        self._state = "idle"

    def get_thinking_open(self) -> str:
        """Return the opening tag for reasoning blocks.

        Returns
        -------
        str
            The reasoning opening tag.
        """
        return THINK_TAG

    def get_thinking_close(self) -> str:
        """Return the closing tag for reasoning blocks.

        Returns
        -------
        str
            The reasoning closing tag.
        """
        return END_TAG

    def parse(self, content: str) -> tuple[str | None, str]:
        """Parse a full response into reasoning and remaining content.

        Parameters
        ----------
        content : str
            The complete model output.

        Returns
        -------
        tuple[str | None, str]
            The reasoning content (if any) and the remaining content.
        """
        content = self._strip_leading_begin_assistant(content)
        if THINK_TAG not in content:
            remaining = self._parse_content_or_calls(content)
            return None, remaining if remaining is not None else content

        think_idx = content.find(THINK_TAG)
        start = think_idx + len(THINK_TAG)
        boundary_idx, boundary_tag = self._find_next_tag(
            content[start:],
            (END_TAG, CONTENT_TAG, TOOL_CALLS_TAG),
        )
        reasoning: str | None
        remaining: str
        if boundary_idx == -1:
            reasoning = content[start:] if start < len(content) else None
            remaining = ""
            return reasoning, remaining

        boundary_idx += start
        reasoning = content[start:boundary_idx]

        remaining = self._parse_content_or_calls(content)
        if remaining is not None:
            return reasoning, remaining

        if boundary_tag == END_TAG:
            remaining = content[boundary_idx + len(END_TAG) :]
        else:
            remaining = content[boundary_idx + len(boundary_tag) :]
        return reasoning, remaining

    def parse_stream(self, chunk: str | None = None) -> tuple[dict[str, Any] | None, bool]:
        """Parse streaming chunks to emit reasoning or content deltas.

        Parameters
        ----------
        chunk : str | None
            The text chunk to parse.

        Returns
        -------
        tuple[dict[str, Any] | None, bool]
            A parsed delta payload and whether reasoning has finished.
        """
        if chunk is None:
            return None, self._state == "done"

        self._buffer += chunk

        while self._buffer:
            if self._state == "idle":
                if self._buffer.startswith(BEGIN_ASSISTANT_TAG):
                    self._buffer = self._buffer[len(BEGIN_ASSISTANT_TAG) :]
                    continue
                pos, tag = self._find_next_tag(
                    self._buffer,
                    (BEGIN_ASSISTANT_TAG, THINK_TAG, CONTENT_TAG, TOOL_CALLS_TAG),
                )
                if pos == -1:
                    return self._emit_safe_content(self._buffer)
                if pos > 0:
                    content = self._buffer[:pos]
                    self._buffer = self._buffer[pos:]
                    return {"content": content}, False

                self._buffer = self._buffer[len(tag) :]
                if tag == BEGIN_ASSISTANT_TAG:
                    continue
                if tag == THINK_TAG:
                    self._state = "thinking"
                    continue

                self._state = "done"
                if self._buffer:
                    content = self._buffer
                    self._buffer = ""
                    return {"content": content}, True
                return None, True

            if self._state == "thinking":
                pos, tag = self._find_next_tag(
                    self._buffer,
                    (END_TAG, CONTENT_TAG, TOOL_CALLS_TAG),
                )
                if pos == -1:
                    return self._emit_safe_reasoning(self._buffer)

                reasoning = self._buffer[:pos]
                remaining = self._buffer[pos + len(tag) :]
                self._buffer = ""
                self._state = "done"
                remaining = self._strip_content_tag(remaining)
                payload: dict[str, Any] = {}
                if reasoning:
                    payload["reasoning_content"] = reasoning
                if remaining:
                    payload["content"] = remaining
                return payload if payload else None, True

            if self._state == "done":
                content = self._buffer
                self._buffer = ""
                return {"content": content}, True

        return None, self._state == "done"

    def _emit_safe_content(self, text: str) -> tuple[dict[str, Any] | None, bool]:
        """Emit content while protecting partial special tags.

        Parameters
        ----------
        text : str
            The pending buffer text.

        Returns
        -------
        tuple[dict[str, Any] | None, bool]
            Content delta and completion flag.
        """
        partial_len = self._max_partial_match(
            text,
            (BEGIN_ASSISTANT_TAG, THINK_TAG, CONTENT_TAG, TOOL_CALLS_TAG),
        )
        if partial_len:
            safe = text[:-partial_len]
            self._buffer = text[-partial_len:]
            if safe:
                return {"content": safe}, False
            return None, False

        self._buffer = ""
        return {"content": text}, False

    def _emit_safe_reasoning(self, text: str) -> tuple[dict[str, Any] | None, bool]:
        """Emit reasoning while protecting partial boundary tags.

        Parameters
        ----------
        text : str
            The pending buffer text.

        Returns
        -------
        tuple[dict[str, Any] | None, bool]
            Reasoning delta and completion flag.
        """
        partial_len = self._max_partial_match(text, (END_TAG, CONTENT_TAG, TOOL_CALLS_TAG))
        if partial_len:
            safe = text[:-partial_len]
            self._buffer = text[-partial_len:]
            if safe:
                return {"reasoning_content": safe}, False
            return None, False

        self._buffer = ""
        return {"reasoning_content": text}, False

    def _parse_content_or_calls(self, text: str) -> str | None:
        """Extract content after the first content/tool_calls tag.

        Parameters
        ----------
        text : str
            The complete model output.

        Returns
        -------
        str | None
            The content after the tag, or None if no tag is found.
        """
        text = self._strip_leading_begin_assistant(text)
        content_idx = text.find(CONTENT_TAG)
        tool_idx = text.find(TOOL_CALLS_TAG)
        if content_idx != -1:
            return text[content_idx + len(CONTENT_TAG) :]
        if tool_idx != -1:
            return text[tool_idx + len(TOOL_CALLS_TAG) :]
        return None

    def _strip_content_tag(self, text: str) -> str:
        """Strip a leading content or tool_calls tag from text.

        Parameters
        ----------
        text : str
            The text to normalize.

        Returns
        -------
        str
            Text without a leading content/tool_calls tag.
        """
        if text.startswith(CONTENT_TAG):
            return text[len(CONTENT_TAG) :]
        if text.startswith(TOOL_CALLS_TAG):
            return text[len(TOOL_CALLS_TAG) :]
        if text.startswith(BEGIN_ASSISTANT_TAG):
            return text[len(BEGIN_ASSISTANT_TAG) :]
        return text

    def _strip_leading_begin_assistant(self, text: str) -> str:
        """Strip a leading assistant header token if present.

        Parameters
        ----------
        text : str
            The text to normalize.

        Returns
        -------
        str
            Text without a leading assistant header token.
        """
        if text.startswith(BEGIN_ASSISTANT_TAG):
            return text[len(BEGIN_ASSISTANT_TAG) :]
        return text

    def _find_next_tag(self, text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """Find the earliest occurrence of any tag.

        Parameters
        ----------
        text : str
            The text to search.
        tags : tuple[str, ...]
            The tags to search for.

        Returns
        -------
        tuple[int, str]
            The position and tag, or (-1, "") if not found.
        """
        earliest = len(text) + 1
        found = ""
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and idx < earliest:
                earliest = idx
                found = tag
        if not found:
            return -1, ""
        return earliest, found

    def _max_partial_match(self, text: str, tags: tuple[str, ...]) -> int:
        """Return the longest partial suffix that matches any tag prefix.

        Parameters
        ----------
        text : str
            The text to inspect.
        tags : tuple[str, ...]
            Tags to test for partial matches.

        Returns
        -------
        int
            The length of the longest partial match.
        """
        max_len = 0
        for tag in tags:
            match_len = self._find_partial_match(text, tag)
            if match_len > max_len:
                max_len = match_len
        return max_len

    def _find_partial_match(self, text: str, tag: str) -> int:
        """Find the longest suffix of text that is a prefix of tag.

        Parameters
        ----------
        text : str
            The text to inspect.
        tag : str
            The tag to match.

        Returns
        -------
        int
            The length of the partial match.
        """
        max_check_len = min(len(text), len(tag))
        for length in range(max_check_len, 0, -1):
            if tag.startswith(text[-length:]):
                return length
        return 0


class SolarOpenToolParser:
    """Parser for Solar Open tool call outputs with streaming support."""

    def __init__(self) -> None:
        """Initialize the tool parser state."""
        self._buffer = ""
        self._mode = "content"
        self._tool_state = "await_begin"
        self._current_tool_name: str | None = None

    def parse(self, content: str) -> tuple[list[dict[str, Any]] | None, str]:
        """Parse a completed response into tool calls and remaining content.

        Parameters
        ----------
        content : str
            The complete model output.

        Returns
        -------
        tuple[list[dict[str, Any]] | None, str]
            Parsed tool calls and remaining content.
        """
        content = self._strip_leading_begin_assistant(content)
        tool_calls = self._parse_tool_calls(content)
        remaining = self._parse_content(content)
        return tool_calls, remaining

    def parse_stream(self, chunk: str | None = None) -> tuple[dict[str, Any] | None, bool]:
        """Parse streaming chunks into content/tool call deltas.

        Parameters
        ----------
        chunk : str | None
            The text chunk to parse.

        Returns
        -------
        tuple[dict[str, Any] | None, bool]
            A parsed delta payload and completion flag.
        """
        if chunk is None:
            return None, True

        self._buffer += chunk

        while self._buffer:
            if self._mode == "content":
                if self._buffer.startswith(BEGIN_ASSISTANT_TAG):
                    self._buffer = self._buffer[len(BEGIN_ASSISTANT_TAG) :]
                    continue
                pos, tag = self._find_next_tag(
                    self._buffer,
                    (
                        BEGIN_ASSISTANT_TAG,
                        TOOL_CALLS_TAG,
                        TOOL_CALL_BEGIN,
                        FLUSH_TAG,
                        END_TAG,
                        CALLS_END,
                    ),
                )
                if pos == -1:
                    return self._emit_safe_tool_content(self._buffer)
                if pos > 0:
                    content = self._buffer[:pos]
                    self._buffer = self._buffer[pos:]
                    return {"content": content}, True

                self._buffer = self._buffer[len(tag) :]
                if tag == BEGIN_ASSISTANT_TAG:
                    continue
                if tag in (TOOL_CALLS_TAG, TOOL_CALL_BEGIN):
                    self._mode = "tool"
                    if tag == TOOL_CALL_BEGIN:
                        self._tool_state = "await_name_tag"
                    continue
                if tag == CALLS_END:
                    self._mode = "content"
                    continue
                continue

            if self._mode == "tool":
                if self._buffer.startswith(CALLS_END):
                    self._buffer = self._buffer[len(CALLS_END) :]
                    self._mode = "content"
                    continue
                if self._buffer.startswith(TOOL_CALLS_TAG):
                    self._buffer = self._buffer[len(TOOL_CALLS_TAG) :]
                    continue
                previous_buffer = self._buffer
                previous_state = self._tool_state
                tool_payload = self._parse_tool_stream()
                if tool_payload is not None:
                    return tool_payload, True
                if self._buffer != previous_buffer or self._tool_state != previous_state:
                    continue
                return None, True

        return None, True

    def _parse_tool_stream(self) -> dict[str, Any] | None:
        """Parse tool call streaming state.

        Returns
        -------
        dict[str, Any] | None
            A tool delta payload when available.
        """
        if self._tool_state == "await_begin":
            pos = self._buffer.find(TOOL_CALL_BEGIN)
            if pos == -1:
                return None
            self._buffer = self._buffer[pos + len(TOOL_CALL_BEGIN) :]
            self._tool_state = "await_name_tag"
            return None

        if self._tool_state == "await_name_tag":
            pos = self._buffer.find(TOOL_CALL_NAME)
            if pos == -1:
                return None
            self._buffer = self._buffer[pos + len(TOOL_CALL_NAME) :]
            self._tool_state = "await_args_tag"
            return None

        if self._tool_state == "await_args_tag":
            pos = self._buffer.find(TOOL_CALL_ARGS)
            if pos == -1:
                return None
            self._current_tool_name = self._buffer[:pos]
            self._buffer = self._buffer[pos + len(TOOL_CALL_ARGS) :]
            self._tool_state = "in_args"
            end_pos = self._buffer.find(TOOL_CALL_END)
            if end_pos != -1:
                arguments = self._buffer[:end_pos]
                self._buffer = self._buffer[end_pos + len(TOOL_CALL_END) :]
                self._tool_state = "await_begin"
                name = self._current_tool_name
                self._current_tool_name = None
                payload: dict[str, Any] = {}
                if name:
                    payload["name"] = name
                if arguments:
                    payload["arguments"] = arguments
                return payload if payload else None
            if self._current_tool_name:
                return {"name": self._current_tool_name}
            return None

        if self._tool_state == "in_args":
            pos = self._buffer.find(TOOL_CALL_END)
            if pos == -1:
                return self._emit_safe_arguments(self._buffer)
            arguments = self._buffer[:pos]
            self._buffer = self._buffer[pos + len(TOOL_CALL_END) :]
            self._tool_state = "await_begin"
            self._current_tool_name = None
            if arguments:
                return {"arguments": arguments}
            return None

        return None

    def _parse_content(self, text: str) -> str:
        """Extract content before tool call markers or terminators.

        Parameters
        ----------
        text : str
            The complete model output.

        Returns
        -------
        str
            Remaining content with tool sections stripped.
        """
        text = self._strip_leading_begin_assistant(text)
        boundary_idx, _ = self._find_next_tag(
            text,
            (BEGIN_ASSISTANT_TAG, TOOL_CALLS_TAG, TOOL_CALL_BEGIN, FLUSH_TAG, END_TAG, CALLS_END),
        )
        if boundary_idx == -1:
            return text
        return text[:boundary_idx]

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from a completed response.

        Parameters
        ----------
        text : str
            The complete model output.

        Returns
        -------
        list[dict[str, Any]]
            Parsed tool call entries.
        """
        tool_calls: list[dict[str, Any]] = []
        text = self._strip_leading_begin_assistant(text)
        section_end = text.find(CALLS_END)
        if section_end == -1:
            section_end = len(text)

        cursor = 0
        while cursor < section_end:
            begin = text.find(TOOL_CALL_BEGIN, cursor, section_end)
            if begin == -1:
                break
            begin += len(TOOL_CALL_BEGIN)
            name = text.find(TOOL_CALL_NAME, begin, section_end)
            if name == -1:
                break
            args = text.find(TOOL_CALL_ARGS, name + len(TOOL_CALL_NAME), section_end)
            if args == -1:
                break
            end = text.find(TOOL_CALL_END, args + len(TOOL_CALL_ARGS), section_end)
            if end == -1:
                break

            tool_name = text[name + len(TOOL_CALL_NAME) : args]
            args_text = text[args + len(TOOL_CALL_ARGS) : end]
            tool_calls.append(
                {
                    "name": tool_name,
                    "arguments": self._parse_tool_call_args(args_text),
                }
            )
            cursor = end + len(TOOL_CALL_END)

        return tool_calls

    def _parse_tool_call_args(self, text: str) -> Any:
        """Parse tool call arguments into a structured object.

        Parameters
        ----------
        text : str
            The arguments text.

        Returns
        -------
        Any
            Parsed arguments or raw text on failure.
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text

    def _emit_safe_tool_content(self, text: str) -> tuple[dict[str, Any] | None, bool]:
        """Emit content while protecting partial tool tags.

        Parameters
        ----------
        text : str
            The pending buffer text.

        Returns
        -------
        tuple[dict[str, Any] | None, bool]
            Content delta and completion flag.
        """
        partial_len = self._max_partial_match(
            text,
            (BEGIN_ASSISTANT_TAG, TOOL_CALLS_TAG, TOOL_CALL_BEGIN, FLUSH_TAG, END_TAG, CALLS_END),
        )
        if partial_len:
            safe = text[:-partial_len]
            self._buffer = text[-partial_len:]
            if safe:
                return {"content": safe}, True
            return None, True

        self._buffer = ""
        return {"content": text}, True

    def _emit_safe_arguments(self, text: str) -> dict[str, Any] | None:
        """Emit argument deltas while protecting partial end tags.

        Parameters
        ----------
        text : str
            The pending argument buffer.

        Returns
        -------
        dict[str, Any] | None
            Argument delta payload if available.
        """
        partial_len = self._max_partial_match(text, (TOOL_CALL_END,))
        if partial_len:
            safe = text[:-partial_len]
            self._buffer = text[-partial_len:]
            if safe:
                return {"arguments": safe}
            return None

        self._buffer = ""
        return {"arguments": text} if text else None

    def _find_next_tag(self, text: str, tags: tuple[str, ...]) -> tuple[int, str]:
        """Find the earliest occurrence of any tag.

        Parameters
        ----------
        text : str
            The text to search.
        tags : tuple[str, ...]
            Tags to search for.

        Returns
        -------
        tuple[int, str]
            The position and tag, or (-1, "") if not found.
        """
        earliest = len(text) + 1
        found = ""
        for tag in tags:
            idx = text.find(tag)
            if idx != -1 and idx < earliest:
                earliest = idx
                found = tag
        if not found:
            return -1, ""
        return earliest, found

    def _max_partial_match(self, text: str, tags: tuple[str, ...]) -> int:
        """Return the longest partial suffix that matches any tag prefix.

        Parameters
        ----------
        text : str
            The text to inspect.
        tags : tuple[str, ...]
            Tags to test for partial matches.

        Returns
        -------
        int
            The length of the longest partial match.
        """
        max_len = 0
        for tag in tags:
            match_len = self._find_partial_match(text, tag)
            if match_len > max_len:
                max_len = match_len
        return max_len

    def _find_partial_match(self, text: str, tag: str) -> int:
        """Find the longest suffix of text that is a prefix of tag.

        Parameters
        ----------
        text : str
            The text to inspect.
        tag : str
            The tag to match.

        Returns
        -------
        int
            The length of the partial match.
        """
        max_check_len = min(len(text), len(tag))
        for length in range(max_check_len, 0, -1):
            if tag.startswith(text[-length:]):
                return length
        return 0

    def _strip_leading_begin_assistant(self, text: str) -> str:
        """Strip a leading assistant header token if present.

        Parameters
        ----------
        text : str
            The text to normalize.

        Returns
        -------
        str
            Text without a leading assistant header token.
        """
        if text.startswith(BEGIN_ASSISTANT_TAG):
            return text[len(BEGIN_ASSISTANT_TAG) :]
        return text
