"""Tests for the Ministral3 tool parser."""

import unittest
from app.handler.parser.ministral3 import Ministral3ToolParser  # noqa: F401

class TestMinistral3ToolParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = Ministral3ToolParser()

    def test_parse_single_tool(self) -> None:
        content = "Some intro [TOOL_CALLS]get_weather[ARGS]{\"city\": \"Seoul\"} more text"
        tool_calls, remaining = self.parser.parse(content)
        self.assertEqual(len(tool_calls), 1)
        expected = {"name": "get_weather", "arguments": {"city": "Seoul"}}
        self.assertEqual(tool_calls[0], expected)
        self.assertEqual(remaining, "more text")

    def test_parse_multiple_tools(self) -> None:
        content = (
            "Start [TOOL_CALLS]first[ARGS]{\"val\": 1} middle "
            "[TOOL_CALLS]second[ARGS]{\"val\": 2} end"
        )
        tool_calls, remaining = self.parser.parse(content)
        self.assertEqual(len(tool_calls), 2)
        expected1 = {"name": "first", "arguments": {"val": 1}}
        expected2 = {"name": "second", "arguments": {"val": 2}}
        self.assertEqual(tool_calls, [expected1, expected2])
        self.assertEqual(remaining, "end")

    def test_parse_stream(self) -> None:
        chunks = [
            "Prefix ",
            "[TOOL_CALLS]stream_tool",
            "[ARGS]{\"x\": \"y\"",
            "}",
        ]
        outputs = []
        for chunk in chunks:
            parsed, complete = self.parser.parse_stream(chunk)
            if parsed is not None:
                outputs.append(parsed)
        final, _ = self.parser.parse_stream(None)
        if final is not None:
            outputs.append(final)

        # Expect four dicts: initial content, two intermediate empty dicts, then name/arguments
        self.assertEqual(len(outputs), 4)
        # First output should contain the prefix content
        self.assertIn("content", outputs[0])
        self.assertEqual(outputs[0]["content"], "Prefix ")
        # Last output should have name and arguments
        self.assertEqual(outputs[-1]["name"], "stream_tool")
        self.assertEqual(outputs[-1]["arguments"], "{\"x\": \"y\"}")
