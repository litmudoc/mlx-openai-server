import json
import re
from typing import Any

from .base import BaseThinkingParser, BaseToolParser

THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class HermesThinkingParser(BaseThinkingParser):
    """Parser for Hermes model's thinking response format."""

    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE,
        )


class HermesToolParser(BaseToolParser):
    """Parser for Hermes model's tool response format.

    Supports two formats:
    1. JSON format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    2. XML format: <tool_call><function=func_name><parameter=arg>value</parameter></function></tool_call>
    """

    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE,
        )
        # Regex patterns for XML-style parsing
        self._function_pattern = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        self._parameter_pattern = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

    def _parse_tool_content(self, tool_content: str) -> dict[str, Any] | None:
        """Parse tool call content, supporting both JSON and XML formats.

        Parameters
        ----------
        tool_content : str
            The content between <tool_call> and </tool_call> tags.

        Returns
        -------
        dict | None
            Parsed tool call with 'name' and 'arguments' keys.
        """
        content = tool_content.strip()

        # Try JSON format first (faster for valid JSON)
        if content.startswith("{"):
            return super()._parse_tool_content(content)

        # Try XML format: <function=name><parameter=arg>value</parameter></function>
        func_match = self._function_pattern.search(content)
        if func_match:
            function_name = func_match.group(1).strip()
            function_body = func_match.group(2).strip()

            # Extract all parameters
            arguments = {}
            for param_match in self._parameter_pattern.finditer(function_body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()

                # Try to parse as JSON value (for numbers, booleans, objects)
                try:
                    arguments[param_name] = json.loads(param_value)
                except json.JSONDecodeError:
                    # Keep as string
                    arguments[param_name] = param_value

            return {"name": function_name, "arguments": arguments}

        # Fallback to parent's JSON parsing
        return super()._parse_tool_content(content)
