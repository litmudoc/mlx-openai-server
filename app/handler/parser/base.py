import json
import logging
from typing import Any

from json_repair import repair_json

logger = logging.getLogger(__name__)


class BaseThinkingParser:
    def __init__(self, thinking_open: str, thinking_close: str):
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        self.is_thinking = False
        self.buffer = ""

    def get_thinking_open(self):
        return self.thinking_open

    def get_thinking_close(self):
        return self.thinking_close

    def parse(self, content: str) -> tuple[str | None, str]:
        start_thinking = content.find(self.thinking_open)
        if start_thinking == -1:
            return None, content

        thinking_open_len = len(self.thinking_open)
        thinking_close_len = len(self.thinking_close)
        start_content = start_thinking + thinking_open_len
        end_thinking = content.find(self.thinking_close, start_content)

        if end_thinking == -1:
            return None, content

        thinking_content = content[start_content:end_thinking].strip()
        remaining_content = content[end_thinking + thinking_close_len :].strip()
        return thinking_content, remaining_content

    def _find_partial_match(self, text: str, tag: str) -> int:
        """Return the length of the longest suffix of 'text' that is a prefix of 'tag'."""
        max_check_len = min(len(text), len(tag))
        for length in range(max_check_len, 0, -1):
            if tag.startswith(text[-length:]):
                return length
        return 0

    def parse_stream(self, chunk: str | None = None) -> tuple[Any | None, bool]:
        """
        Parse streaming chunks for thinking content.

        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if thinking section is complete
        """
        if chunk:
            self.buffer += chunk

        result = None
        is_complete = False

        while self.buffer:
            if not self.is_thinking:
                # Look for opening tag
                start_idx = self.buffer.find(self.thinking_open)
                if start_idx != -1:
                    # Found opening tag
                    # Yield content before tag (if any)
                    if start_idx > 0:
                        pre_content = self.buffer[:start_idx]
                        # We need to return this. But we can only return one thing.
                        # If we have pre-content, return it now, and keep the rest for next call?
                        # Or return tuple? The contract expects one item.
                        # We will prioritize returning pre-content.
                        # Remove pre-content from buffer
                        self.buffer = self.buffer[start_idx:]
                        return pre_content, False

                    # Buffer starts with thinking_open. Switch state.
                    self.is_thinking = True
                    self.buffer = self.buffer[len(self.thinking_open) :]
                    continue
                # No opening tag found. Check for partial match at end.
                partial_len = self._find_partial_match(self.buffer, self.thinking_open)
                if partial_len > 0:
                    # Yield safe part
                    safe_len = len(self.buffer) - partial_len
                    if safe_len > 0:
                        content = self.buffer[:safe_len]
                        self.buffer = self.buffer[safe_len:]
                        return content, False
                    # Entire buffer is a partial match, wait for more data
                    return None, False
                # No partial match, yield everything
                content = self.buffer
                self.buffer = ""
                return content, False
            # Inside thinking block
            # Look for closing tag
            end_idx = self.buffer.find(self.thinking_close)
            if end_idx != -1:
                # Found closing tag
                reasoning_content = self.buffer[:end_idx]
                self.buffer = self.buffer[end_idx + len(self.thinking_close) :]
                self.is_thinking = False

                # Yield reasoning content and signal completion
                # If buffer has more content (after close), we leave it in buffer.
                # Caller handles is_complete=True by disabling parser, so leftover buffer
                # should be returned?
                # If is_complete=True, the handler sets thinking_parser=None.
                # The handler needs to handle the rest of the text.
                # But parse_stream is called one last time?
                # The handler loop:
                # parsed, complete = parser.parse_stream(text)
                # if parsed: yield
                # if complete: parser = None
                # if after_thinking_close_content: text = that

                # We can use the dictionary return to pass extra content if needed,
                # but current BaseThinkingParser logic in handler supports:
                # if parsed is dict: after_thinking_close_content = parsed.pop("content")

                res = {"reasoning_content": reasoning_content}
                if self.buffer:
                    res["content"] = self.buffer
                    self.buffer = ""

                return res, True
            # No closing tag. Check partial match.
            partial_len = self._find_partial_match(self.buffer, self.thinking_close)
            if partial_len > 0:
                # Yield safe reasoning
                safe_len = len(self.buffer) - partial_len
                if safe_len > 0:
                    reasoning = self.buffer[:safe_len]
                    self.buffer = self.buffer[safe_len:]
                    return {"reasoning_content": reasoning}, False
                return None, False
            # Yield all as reasoning
            reasoning = self.buffer
            self.buffer = ""
            return {"reasoning_content": reasoning}, False

        return None, is_complete


class ParseToolState:
    NORMAL = 0
    FOUND_PREFIX = 1


class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: str | None = None):
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""
        self.state = ParseToolState.NORMAL
        # Pre-calculate lengths for performance
        self._tool_open_len = len(tool_open)
        self._tool_close_len = len(tool_close) if tool_close else 0

    def get_tool_open(self):
        return self.tool_open

    def get_tool_close(self):
        return self.tool_close

    def _set_content(self, res: dict[str, Any], content: str) -> None:
        """Helper to set content only if non-empty."""
        res["content"] = content if content else None

    def _parse_tool_content(self, tool_content: str) -> dict[str, Any] | None:
        """
        Parses the content of a tool call. Subclasses can override this method
        to support different content formats (e.g., XML, YAML).
        Args:
            tool_content: The string content extracted from between the tool tags.

        Returns:
            A dictionary representing the parsed tool call, or None if parsing fails.
        """

        try:
            repaired_json = repair_json(tool_content)
            return json.loads(repaired_json)
        except json.JSONDecodeError:
            raise

    def parse(self, content: str) -> tuple[list[dict[str, Any]] | None, str]:
        tool_calls = []
        remaining_parts = []

        if self.tool_open not in content:
            return [], content

        tool_open_len = len(self.tool_open)
        tool_close_len = len(self.tool_close)
        pos = 0

        while True:
            start_tool = content.find(self.tool_open, pos)
            if start_tool == -1:
                # No more tool calls, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break

            # Add content before tool call
            if start_tool > pos:
                remaining_parts.append(content[pos:start_tool].strip())

            # Find closing tag
            search_start = start_tool + tool_open_len
            end_tool = content.find(self.tool_close, search_start)
            if end_tool == -1:
                # Unclosed tool tag, add remaining content and break
                remaining_parts.append(content[pos:].strip())
                break

            # Extract and parse tool content
            tool_content = content[search_start:end_tool].strip()
            try:
                json_output = self._parse_tool_content(tool_content)
                tool_calls.append(json_output)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                # Continue processing remaining content after error
                remaining_parts.append(content[pos:].strip())
                break

            # Move position past the closing tag
            pos = end_tool + tool_close_len

        remaining_content = " ".join(filter(None, remaining_parts))
        return tool_calls, remaining_content

    def _find_partial_match(self, text: str, tag: str) -> int:
        """Return the length of the longest suffix of 'text' that is a prefix of 'tag'."""
        max_check_len = min(len(text), len(tag))
        for length in range(max_check_len, 0, -1):
            if tag.startswith(text[-length:]):
                return length
        return 0

    def parse_stream(self, chunk: str | None = None) -> tuple[dict[str, Any] | None, bool]:
        """
        Parse streaming chunks for tool calls.
        Args:
            chunk: The text chunk to parse, or None for empty chunks
        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed chunk (could be str, dict)
                - is_complete: True if item is ready to yield
        """
        if chunk:
            self.buffer += chunk

        res = {}

        while self.buffer:
            if self.state == ParseToolState.NORMAL:
                # Look for tool_open
                start_idx = self.buffer.find(self.tool_open)
                if start_idx != -1:
                    # Found start
                    # Yield content before tool
                    content = self.buffer[:start_idx]
                    self.buffer = self.buffer[start_idx + self._tool_open_len :]
                    self.state = ParseToolState.FOUND_PREFIX

                    if content:
                        self._set_content(res, content)
                        return res, True
                    # If no content, continue loop to process tool immediately
                    continue
                # Check partial match
                partial_len = self._find_partial_match(self.buffer, self.tool_open)
                if partial_len > 0:
                    # Yield safe part
                    safe_len = len(self.buffer) - partial_len
                    if safe_len > 0:
                        content = self.buffer[:safe_len]
                        self.buffer = self.buffer[safe_len:]
                        self._set_content(res, content)
                        return res, True
                    # Buffer is all partial, wait
                    return None, True  # Using True to signal "no error, just waiting"?
                    # Wait, handler usage: `if is_complete: yield parsed`.
                    # So if we return None, True -> yields None (filtered out).
                    # Correct.
                    return None, True
                # Yield all
                content = self.buffer
                self.buffer = ""
                self._set_content(res, content)
                return res, True

            if self.state == ParseToolState.FOUND_PREFIX:
                # Look for tool_close
                end_idx = self.buffer.find(self.tool_close)
                if end_idx != -1:
                    # Found end
                    tool_content = self.buffer[:end_idx]
                    self.buffer = self.buffer[end_idx + self._tool_close_len :]
                    self.state = ParseToolState.NORMAL

                    try:
                        json_output = self._parse_tool_content(tool_content)
                        res["name"] = str(json_output["name"])
                        # arguments must be a JSON string, not str(dict)
                        args = json_output["arguments"]
                        if isinstance(args, dict):
                            res["arguments"] = json.dumps(args)
                        else:
                            res["arguments"] = str(args)
                        return res, True
                    except Exception as e:
                        logger.error(f"Error parsing tool call: {e}")
                        # If parsing fails, what do we do? Yield raw content?
                        # Recover by yielding raw content + close tag?
                        # For now, just log and yield nothing (swallow bad tool call) or yield as text?
                        # Let's yield as text to be safe
                        self._set_content(res, self.tool_open + tool_content + self.tool_close)
                        return res, True
                else:
                    # Check partial match
                    partial_len = self._find_partial_match(self.buffer, self.tool_close)
                    if partial_len > 0:
                        # We are buffering tool content. Do NOT yield partials.
                        # Wait for full close tag.
                        return None, True
                    # Still buffering.
                    return None, True

        return None, True


"""
Base Message Converter
Provides generic conversion from OpenAI API message format to model-compatible format.
"""


class BaseMessageConverter:
    """Base message format converter class"""

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert message format to be compatible with specific model chat templates"""
        converted_messages = []

        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)

        return converted_messages

    def _convert_single_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert a single message"""
        if not isinstance(message, dict):
            return message

        # Convert function.arguments from string to object in tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            self._convert_tool_calls(tool_calls)

        return message

    def _convert_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Convert arguments format in tool calls"""
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                if isinstance(function, dict) and "arguments" in function:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        function["arguments"] = self._parse_arguments_string(arguments)

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse arguments string to object, can be overridden by subclasses"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str
