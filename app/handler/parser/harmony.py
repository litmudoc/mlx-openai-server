from enum import Enum
from functools import lru_cache
import logging
from typing import Any

from openai_harmony import (
    HarmonyEncodingName,
    Role,
    StreamState,
    StreamableParser,
    load_harmony_encoding,
)

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Enumeration of harmony channel types."""

    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


class ParsingState(Enum):
    """Enumeration of parsing states."""

    IDLE = "idle"
    PROCESSING_TOKENS = "processing_tokens"
    TOOL_PARSING = "tool_parsing"
    STREAM_ENDED = "stream_ended"


# Harmony Parsing Helper Functions
@lru_cache(maxsize=2)
def get_harmony_stop_sequences(include_message_end: bool = True) -> list[str]:
    """Return Harmony stop sequences for GPT-OSS completions.

    Parameters
    ----------
    include_message_end : bool
        Whether to include <|end|> (message terminator) as a stop sequence.

    Returns
    -------
    list[str]
        Stop sequences decoded from the Harmony encoding.
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    if include_message_end:
        token_ids = enc.stop_tokens()
    else:
        token_ids = enc.stop_tokens_for_assistant_actions()
    return [enc.decode([token_id]) for token_id in token_ids]


class HarmonyParser:
    """
    Enhanced helper class for parsing GPT-OSS model responses using harmony encoding.

    This parser handles streaming and non-streaming responses with proper state management,
    error handling, and support for different harmony channels (analysis, commentary, final).
    """

    def __init__(self):
        """Initialize the harmony parser with encoding and state management."""
        try:
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.parser = StreamableParser(self.enc, role=Role.ASSISTANT)
        except Exception as e:
            logger.error(f"Failed to initialize harmony encoding: {e}")
            raise

        # Configuration
        self.end_tool_chunk = "<|call|>"

        # State management
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset the parser state to initial values."""
        self.tool_state = False
        self.end_stream = False
        self.parsing_state = ParsingState.IDLE
        self._accumulated_content = {
            ChannelType.ANALYSIS.value: [],
            ChannelType.COMMENTARY.value: [],
            ChannelType.FINAL.value: [],
        }
        self._current_function_name = None
        self._function_arguments = []

    def _reset_stream_parser(self) -> None:
        """Reset the stream parser to a fresh state for resynchronization."""
        self.parser = StreamableParser(self.enc, role=Role.ASSISTANT)

    def _should_resync(self, token_error: Exception) -> bool:
        """Check whether a token parsing error should trigger a resync."""
        message = str(token_error)
        return "expecting start token" in message or "message header" in message

    def parse_stream(self, text: str | None = None) -> tuple[Any | None, bool]:
        """
        Parse streaming text input and return parsing state and extracted content.

        Args:
            text: The text chunk to parse, or None for empty chunks

        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if stream has ended

        Raises:
            Exception: If encoding or parsing fails
        """
        # Handle end of stream marker without skipping parser state updates
        if text == self.end_tool_chunk:
            logger.debug("End tool chunk detected, marking stream as ended")
            self.end_stream = True
            self.parsing_state = ParsingState.STREAM_ENDED

        # Handle empty or None text
        if not text:
            return None, self.end_stream

        try:
            self.parsing_state = ParsingState.PROCESSING_TOKENS
            if self.parser.state in {StreamState.EXPECT_START, StreamState.HEADER}:
                if "<|start|>" in text:
                    start_index = text.find("<|start|>")
                    if start_index > 0:
                        logger.debug("Dropping %d chars before harmony start token", start_index)
                        text = text[start_index:]
                else:
                    tool_index = text.find(" to=functions.")
                    channel_index = text.find("<|channel|>")
                    candidates = [idx for idx in (tool_index, channel_index) if idx >= 0]
                    if candidates:
                        start_at = min(candidates)
                        if start_at > 0:
                            logger.debug(
                                "Dropping %d chars before harmony content token", start_at
                            )
                        text = text[start_at:]
            text_tokens = self.enc.encode(text, allowed_special="all")

            # Initialize local variables for this chunk
            contents: list[str] = []
            function_name: str | None = None
            function_arguments: list[str] = []
            reasoning_content: list[str] = []
            current_channel: str | None = None
            last_content_channel: str | None = None

            # Process each token
            for text_token in text_tokens:
                for attempt in range(2):
                    try:
                        stream_text = self.parser.process(text_token)
                        current_channel = stream_text.current_channel
                        content = stream_text.last_content_delta

                        if not content:
                            break
                        last_content_channel = current_channel

                        # Handle different channels
                        if current_channel == ChannelType.ANALYSIS.value:
                            reasoning_content.append(content)
                            self._accumulated_content[ChannelType.ANALYSIS.value].append(content)

                        elif current_channel == ChannelType.COMMENTARY.value:
                            self.parsing_state = ParsingState.TOOL_PARSING

                            if self.tool_state:
                                # Already parsing function arguments
                                function_arguments.append(content)
                                self._function_arguments.append(content)
                            else:
                                # Start of new function call
                                self.tool_state = True
                                if (
                                    hasattr(stream_text, "current_recipient")
                                    and stream_text.current_recipient
                                ):
                                    function_name = stream_text.current_recipient.replace(
                                        "functions.", ""
                                    )
                                    self._current_function_name = function_name
                                function_arguments = [content]
                                self._function_arguments = [content]

                        elif current_channel == ChannelType.FINAL.value:
                            contents.append(content)
                            self._accumulated_content[ChannelType.FINAL.value].append(content)
                        break

                    except Exception as token_error:
                        if attempt == 0 and self._should_resync(token_error):
                            logger.debug(
                                "Resynchronizing harmony stream after token error: %s",
                                token_error,
                            )
                            self._reset_stream_parser()
                            continue
                        logger.warning(f"Error processing token {text_token}: {token_error}")
                        break

            # Return appropriate response based on current channel
            resolved_channel = current_channel or last_content_channel
            return self._build_response(
                resolved_channel,
                {
                    "reasoning_content": reasoning_content,
                    "function_name": function_name,
                    "function_arguments": function_arguments,
                    "contents": contents,
                },
            )

        except Exception as e:
            logger.error(f"Error in parse_stream: {e}")
            return None, self.end_stream

    def _build_response(
        self, current_channel: str | None, content_data: dict[str, Any]
    ) -> tuple[dict[str, Any] | str | None, bool]:
        """
        Build the appropriate response based on the current channel.

        Args:
            current_channel: The current harmony channel being processed
            content_data: Dictionary containing extracted content from different sources

        Returns:
            Tuple[parsed_content, is_complete]:
                - parsed_content: The parsed content (str or dict)
                - is_complete: Whether the stream has ended
        """
        if not current_channel:
            return None, self.end_stream

        try:
            if current_channel == ChannelType.ANALYSIS.value:
                reasoning_content = content_data.get("reasoning_content", [])
                if reasoning_content:
                    return {"reasoning_content": "".join(reasoning_content)}, self.end_stream

            elif current_channel == ChannelType.COMMENTARY.value:
                function_name = content_data.get("function_name")
                function_arguments = content_data.get("function_arguments", [])

                response = {}
                if function_name:
                    response["name"] = function_name
                if function_arguments:
                    response["arguments"] = "".join(function_arguments)

                if response:
                    return response, self.end_stream

            elif current_channel == ChannelType.FINAL.value:
                contents = content_data.get("contents", [])
                if contents:
                    return "".join(contents), self.end_stream
        except Exception as e:
            logger.error(f"Error building response for channel {current_channel}: {e}")

        return None, self.end_stream

    def reset(self) -> None:
        """Reset the parser to initial state for reuse."""
        logger.debug("Resetting harmony parser state")
        self._reset_state()

    def get_accumulated_content(self, channel: str | None = None) -> dict[str, str]:
        """
        Get accumulated content for all channels or a specific channel.

        Args:
            channel: Optional specific channel to retrieve content for

        Returns:
            Dictionary of channel content
        """
        if channel and channel in self._accumulated_content:
            return {channel: "".join(self._accumulated_content[channel])}

        return {
            ch: "".join(content) for ch, content in self._accumulated_content.items() if content
        }

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse complete text response and extract structured content.

        This method processes the entire text at once (non-streaming) and extracts
        reasoning content, tool calls, and final content based on harmony channels.

        Args:
            text: The complete text response to parse

        Returns:
            Dictionary containing parsed content with keys:
            - reasoning_content: Analysis/thinking content
            - tool_calls: List of tool call objects
            - content: Final response content

        Raises:
            Exception: If encoding or parsing fails
        """
        # Initialize result structure
        result = {"reasoning_content": None, "tool_calls": None, "content": None}

        if not text:
            logger.warning("Empty text provided to parse method")
            return result

        try:
            # Remove end tool chunk if present
            clean_text = text
            if self.end_tool_chunk in text:
                clean_text = text.split(self.end_tool_chunk)[0]
                logger.debug(f"Removed end tool chunk, processing {len(clean_text)} characters")

            # Encode and parse messages
            tokens = self.enc.encode(clean_text, allowed_special="all")
            parsed_messages = self.enc.parse_messages_from_completion_tokens(
                tokens, role=Role.ASSISTANT
            )

            # Process each parsed message
            for message in parsed_messages:
                try:
                    if not hasattr(message, "channel") or not hasattr(message, "content"):
                        logger.warning(f"Invalid message structure: {message}")
                        continue

                    if message.channel == ChannelType.ANALYSIS.value:
                        if message.content and len(message.content) > 0:
                            result["reasoning_content"] = message.content[0].text
                            logger.debug("Extracted reasoning content")

                    elif message.channel == ChannelType.COMMENTARY.value:
                        if (
                            hasattr(message, "recipient")
                            and message.recipient
                            and message.content
                            and len(message.content) > 0
                        ):
                            tool_call = {
                                "name": message.recipient.replace("functions.", ""),
                                "arguments": message.content[0].text,
                            }
                            result["tool_calls"] = [tool_call]
                            logger.debug(f"Extracted tool call: {tool_call['name']}")

                    elif message.channel == ChannelType.FINAL.value:
                        if message.content and len(message.content) > 0:
                            result["content"] = message.content[0].text
                            logger.debug("Extracted final content")

                except Exception as msg_error:
                    logger.warning(f"Error processing message: {msg_error}")
                    continue

        except Exception as e:
            logger.error(f"Error in parse method: {e}")
            # Return partial results if available, don't raise

        return result

    def is_stream_ended(self) -> bool:
        """Check if the stream has ended."""
        return self.end_stream

    def get_parsing_state(self) -> ParsingState:
        """Get the current parsing state."""
        return self.parsing_state

    def is_tool_parsing_active(self) -> bool:
        """Check if currently parsing tool calls."""
        return self.tool_state

    def get_current_function_info(self) -> dict[str, str | None]:
        """
        Get information about the currently parsed function.

        Returns:
            Dictionary with function name and accumulated arguments
        """
        return {
            "name": self._current_function_name,
            "arguments": "".join(self._function_arguments) if self._function_arguments else None,
        }

    def __repr__(self) -> str:
        """String representation of the parser state."""
        return (
            f"HarmonyParser(state={self.parsing_state.value}, "
            f"tool_state={self.tool_state}, "
            f"stream_ended={self.end_stream})"
        )
