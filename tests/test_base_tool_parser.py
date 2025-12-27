import pytest
from app.handler.parser.base import BaseToolParser

class ConcreteToolParser(BaseToolParser):
    def __init__(self):
        super().__init__(tool_open="<tool>", tool_close="</tool>")

def test_parse_stream_normal_text():
    parser = ConcreteToolParser()
    chunk = "Hello world"
    res, is_complete = parser.parse_stream(chunk)
    
    assert is_complete is True
    assert res == {"content": "Hello world"}
    assert "name" not in res
    assert "arguments" not in res

def test_parse_stream_tool_call():
    parser = ConcreteToolParser()
    
    # 1. Start of tool call
    chunk1 = "Some text <tool>"
    res1, is_complete1 = parser.parse_stream(chunk1)
    assert is_complete1 is True
    assert res1 == {"content": "Some text "}
    
    # 2. Tool call content
    chunk2 = '{"name": "test_tool", "arguments": {}}'
    res2, is_complete2 = parser.parse_stream(chunk2)
    assert is_complete2 is True # Buffering return None, True
    assert res2 is None
    
    # 3. End of tool call
    chunk3 = "</tool> trailing"
    res3, is_complete3 = parser.parse_stream(chunk3)
    assert is_complete3 is True
    assert res3["name"] == "test_tool"
    assert res3["arguments"] == "{}"
    # Content is buffered
    assert res3.get("content") is None
    
    # 4. Flush trailing content
    res4, is_complete4 = parser.parse_stream(None)
    assert is_complete4 is True
    assert res4["content"] == " trailing"

def test_parse_stream_split_tags():
    parser = ConcreteToolParser()
    
    # Split opening tag: "<to" + "ol>"
    res1, _ = parser.parse_stream("Prefix <to")
    # Should yield "Prefix " and buffer "<to"
    assert res1 == {"content": "Prefix "}
    
    res2, _ = parser.parse_stream("ol> content")
    # Buffer: "<to" + "ol> content" = "<tool> content"
    # Found tool_open.
    # Switch to FOUND_PREFIX.
    # Buffer: " content".
    # Return None, True. Buffer " content".
    assert res2 is None
    
    res3, _ = parser.parse_stream("</tool>")
    # Buffer " content</tool>"
    # Found tool_close.
    # Parse " content". (Assuming it's valid tool content json, mock it)
    # Return tool call.
    # BaseToolParser swallows error and returns text if json fails.
    assert res3 is not None
    assert "content" in res3
    assert "<tool> content</tool>" in res3["content"]

def test_parse_stream_empty_chunk():
    parser = ConcreteToolParser()
    res, is_complete = parser.parse_stream(None)
    assert res is None
    assert is_complete is True