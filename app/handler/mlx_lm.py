"""MLX LM Handler for text-only language model requests.

This module provides a handler for making requests to MLX text-only language
models with KVCache reuse support for improved performance.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import gc
from http import HTTPStatus
import time
from typing import Any
import uuid

from fastapi import HTTPException
from loguru import logger
from mlx_lm.models.cache import make_prompt_cache

from ..core.kv_cache_manager import KVCacheManager
from ..core.queue import RequestQueue
from ..core.service_llm_engine import GenerationContext
from ..models.mlx_lm import MLX_LM
from ..schemas.openai import ChatCompletionRequest, EmbeddingRequest, UsageInfo
from ..utils.errors import create_error_response
from .parser import ParserFactory


class MLXLMHandler:
    """Handler for MLX text-only language model requests.

    Provides request queuing, KVCache reuse, metrics tracking, and robust
    error handling for text generation requests.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 32768,
        max_concurrency: int = 1,
        max_prompt_cache: int = 4,
        cache_min_prefix_length: int = 10,
        cache_min_reuse_ratio: float = 0.25,
        enable_auto_tool_choice: bool = False,
        tool_call_parser: str | None = None,
        reasoning_parser: str | None = None,
        trust_remote_code: bool = False,
        chat_template_file: str | None = None,
    ) -> None:
        """Initialize the handler with the specified model path.

        Parameters
        ----------
        model_path : str
            Path to the model directory.
        context_length : int
            Maximum context length for the model.
        max_concurrency : int
            Maximum number of concurrent model inference tasks.
        max_prompt_cache : int
            Maximum number of cached prompts to store for reuse.
        cache_min_prefix_length : int
            Minimum prefix length required for cache reuse.
        cache_min_reuse_ratio : float
            Minimum reuse ratio (prefix_len / cached_tokens_len) required for cache reuse.
            Prevents reusing caches with very low reuse ratios. Defaults to 0.25 (25%).
        enable_auto_tool_choice : bool
            Enable automatic tool choice.
        tool_call_parser : str | None
            Name of the tool call parser to use.
        reasoning_parser : str | None
            Name of the reasoning parser to use.
        trust_remote_code : bool
            Enable trust_remote_code when loading models.
        chat_template_file : str | None
            Path to a custom chat template file.
        """
        self.model_path = model_path
        self.model = MLX_LM(
            model_path,
            context_length,
            trust_remote_code=trust_remote_code,
            chat_template_file=chat_template_file,
        )
        self.model_created = int(time.time())
        self.model_type = self.model.get_model_type()

        # Store parser configuration
        self.enable_auto_tool_choice = enable_auto_tool_choice
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser

        # Initialize request queue for text tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        # Initialize KVCacheManager for prompt cache reuse
        self.cache_manager = KVCacheManager(
            max_cache_count=max_prompt_cache,
            min_prefix_length=cache_min_prefix_length,
            min_reuse_ratio=cache_min_reuse_ratio,
        )

        # Initialize message converter for supported models
        self.converter = ParserFactory.create_converter(self.model_type)

        logger.info(
            f"Initialized MLXLMHandler with model path: {model_path}, "
            f"max_prompt_cache={max_prompt_cache}"
        )

    def _create_parsers(self) -> tuple[Any | None, Any | None]:
        """
        Create appropriate parsers based on model type and available tools.
        Uses ParserFactory for centralized parser creation logic.

        Returns:
            Tuple of (thinking_parser, tool_parser)
        """

        return ParserFactory.create_parsers(
            model_type=self.model_type,
            manual_reasoning_parser=self.reasoning_parser,
            manual_tool_parser=self.tool_call_parser,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        if not text:
            return 0
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def _convert_tool_calls_for_template(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert tool_calls arguments from JSON string to dict for chat template.

        Some chat templates (e.g., Qwen3) expect tool_calls[].function.arguments
        to be a dict, but OpenAI API spec uses JSON strings. This method converts
        the arguments to dict format for template compatibility.

        Parameters
        ----------
        messages : list[dict]
            List of messages that may contain tool_calls.

        Returns
        -------
        list[dict]
            Messages with tool_calls arguments converted to dict.
        """
        import json

        converted = []
        for msg in messages:
            if not isinstance(msg, dict):
                converted.append(msg)
                continue

            # Check if message has tool_calls
            tool_calls = msg.get("tool_calls")
            if not tool_calls or not isinstance(tool_calls, list):
                converted.append(msg)
                continue

            # Deep copy and convert
            new_msg = msg.copy()
            new_tool_calls = []

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    new_tool_calls.append(tc)
                    continue

                new_tc = tc.copy()
                function = tc.get("function")

                if function and isinstance(function, dict):
                    new_function = function.copy()
                    args = function.get("arguments")

                    # Convert string arguments to dict
                    if isinstance(args, str):
                        try:
                            new_function["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            # Keep as string if parsing fails
                            pass

                    new_tc["function"] = new_function

                new_tool_calls.append(new_tc)

            new_msg["tool_calls"] = new_tool_calls
            converted.append(new_msg)

        return converted

    def _count_message_tokens(self, messages: list[dict[str, str]], **kwargs) -> int:
        """
        Count the number of tokens in a list of messages after applying chat template.

        Args:
            messages: List of messages to count tokens for.
            **kwargs: Additional arguments to pass to apply_chat_template.

        Returns:
            int: The number of prompt tokens.
        """
        try:
            input_tokens = self.model.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs
            )
            # Extract token IDs from BatchEncoding if necessary
            if hasattr(input_tokens, "input_ids"):
                input_tokens = input_tokens.input_ids
            return len(input_tokens)
        except Exception as e:
            if "tool_choice" in kwargs and isinstance(kwargs["tool_choice"], str):
                logger.debug(f"Tool choice is string: {kwargs['tool_choice']}")
            logger.warning(f"Failed to count message tokens: {e!s}")
            # Fallback: rough estimate
            total_text = " ".join(
                [
                    msg.get("content", "")
                    for msg in messages
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str)
                ]
            )
            return self._count_tokens(total_text)

    def _extract_model_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from the loaded MLX model.

        Returns:
            dict: Metadata about the model including context_length, backend, etc.
        """
        metadata = {
            "backend": "mlx",
            "modality": "text",
        }

        # Add context length
        if hasattr(self.model, "max_kv_size"):
            metadata["context_length"] = self.model.max_kv_size

        # Add vocab size if available
        if hasattr(self.model.tokenizer, "vocab_size"):
            metadata["vocab_size"] = self.model.tokenizer.vocab_size

        # Add model family/type if available
        if hasattr(self.model, "model_type") and self.model.model_type:
            metadata["model_family"] = self.model.model_type

        # Add dtype info if available from model config
        if hasattr(self.model.model, "dtype"):
            metadata["dtype"] = str(self.model.model.dtype)

        # Add load settings
        metadata["load_settings"] = {"model_path": self.model_path, "model_type": "lm"}

        return metadata

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                    "metadata": self._extract_model_metadata(),
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models: {e!s}")
            return []

    async def initialize(self, queue_config: dict[str, Any] | None = None):
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {"max_concurrency": 1, "timeout": 300, "queue_size": 100}
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency"),
            timeout=queue_config.get("timeout"),
            queue_size=queue_config.get("queue_size"),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXHandler and started request queue")

    def _run_model_sync(
        self,
        loop: asyncio.AbstractEventLoop,
        request_data: dict[str, Any],
        stream_queue: asyncio.Queue | None = None,
    ) -> Any:
        """Run model generation in a separate thread."""
        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                result = self.model.get_embeddings(request_data["input"])
                gc.collect()
                return result

            # Extract request parameters
            messages = request_data.get("messages", [])
            stream = request_data.get("stream", False)
            context = request_data.get("context")

            # Remove these keys from model_params
            model_params = request_data.copy()
            model_params.pop("messages", None)
            model_params.pop("stream", None)
            model_params.pop("context", None)
            model_params.pop("stream_queue", None)
            # Remove internal keys
            model_params.pop("_cached_kv", None)
            model_params.pop("_input_tokens", None)
            model_params.pop("_entry_id", None)

            # Helper to put items in queue safely
            def put_in_queue(item):
                if stream_queue:
                    loop.call_soon_threadsafe(stream_queue.put_nowait, item)

            # Apply message conversion if needed
            if self.converter:
                refined_messages = self.converter.convert_messages(messages)
            else:
                refined_messages = [
                    {k: v for k, v in msg.items() if v is not None} for msg in messages
                ]

            # Use cached_kv passed from _process_request
            cached_kv = request_data.get("_cached_kv")
            input_tokens = request_data.get(
                "_input_tokens"
            )  # This might be None if cache lookup failed or wasn't done?
            entry_id = request_data.get("_entry_id")

            # Fallback if input_tokens not provided (shouldn't happen with updated logic)
            if input_tokens is None:
                # Re-tokenize
                chat_template_kwargs = model_params.get("chat_template_kwargs", {})
                input_tokens = self.model.tokenizer.apply_chat_template(
                    refined_messages,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
                # Extract token IDs from BatchEncoding if necessary
                if hasattr(input_tokens, "input_ids"):
                    input_tokens = input_tokens.input_ids

            # Call the model with cache
            response, prompt_tokens, cache = self.model(
                messages=refined_messages,
                stream=stream,
                prompt_cache=cached_kv,
                context=context,
                **model_params,
            )

            if stream and stream_queue:
                # Iterate generator
                generated_tokens = []

                try:
                    for chunk in response:
                        if chunk:  # Check chunk validity
                            text = chunk if isinstance(chunk, str) else chunk.text
                            if text:
                                chunk_tokens = self.model.tokenizer.encode(
                                    text, add_special_tokens=False
                                )
                                generated_tokens.extend(chunk_tokens)
                            put_in_queue(chunk)

                    # Done streaming
                    # We need to save cache.
                    # `cache_manager.save_cache` is async.
                    # Schedule it on the main loop.
                    async def save_cache_async():
                        try:
                            full_token_ids = input_tokens + generated_tokens
                            # We need access to self.cache_manager which is async
                            await self.cache_manager.save_cache(
                                cache=cache,
                                token_ids=full_token_ids,
                                entry_id=entry_id,
                            )
                            logger.info(f"Cache saved (streaming): {len(full_token_ids)} tokens")
                        except Exception as e:
                            logger.warning(f"Failed to save cache: {e}")
                            await self.cache_manager.unlock_entry(entry_id)

                    future = asyncio.run_coroutine_threadsafe(save_cache_async(), loop)

                    # Signal done
                    put_in_queue(None)

                except Exception as e:
                    put_in_queue(e)
                    # Also unlock cache if failed
                    if entry_id is not None:
                        asyncio.run_coroutine_threadsafe(
                            self.cache_manager.unlock_entry(entry_id), loop
                        )

            else:
                # Non-streaming
                # Tokenize response
                response_text = (
                    response if isinstance(response, str) else response.get("content", "")
                )
                generated_tokens = self.model.tokenizer.encode(
                    response_text, add_special_tokens=False
                )
                full_token_ids = input_tokens + generated_tokens

                # Save cache async
                async def save_cache_sync_async():
                    try:
                        await self.cache_manager.save_cache(
                            cache=cache,
                            token_ids=full_token_ids,
                            entry_id=entry_id,
                        )
                    except Exception:
                        if entry_id is not None:
                            await self.cache_manager.unlock_entry(entry_id)

                asyncio.run_coroutine_threadsafe(save_cache_sync_async(), loop)

                gc.collect()
                return response, prompt_tokens

        except Exception as e:
            if stream_queue:
                put_in_queue(e)
            raise e

    async def generate_text_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Yields:
            str or dict: Response chunks (str) followed by usage info (dict) at the end.
        """
        request_id = f"text-{uuid.uuid4()}"
        context = GenerationContext()
        stream_queue = asyncio.Queue()

        try:
            chat_messages, model_params = await self._prepare_text_request(request)

            # Count prompt tokens
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})
            prompt_tokens = self._count_message_tokens(chat_messages, **chat_template_kwargs)

            request_data = {
                "messages": chat_messages,
                "stream": True,
                "context": context,
                "stream_queue": stream_queue,
                **model_params,
            }

            # Enqueue request - this adds to queue but doesn't block for result
            await self.request_queue.enqueue(request_id, request_data)

            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                enable_thinking = chat_template_kwargs.get("enable_thinking", True)
                if not enable_thinking:
                    thinking_parser = None

            is_first_chunk = True
            completion_chunks = []
            after_thinking_close_content = None

            # Consume queue
            while True:
                item = await stream_queue.get()

                if item is None:
                    break

                if isinstance(item, Exception):
                    raise item

                # MLX_LM yields object with .text usually, or str?
                # engine yields str. MLX_LM yields engine generator (str).
                chunk_text = item
                if not isinstance(item, str) and hasattr(item, "text"):
                    chunk_text = item.text

                if not chunk_text:
                    continue

                completion_chunks.append(chunk_text)
                text = chunk_text

                if thinking_parser and ParserFactory.has_special_parsing(self.reasoning_parser):
                    parsed_content, is_complete = thinking_parser.parse_stream(text)
                    if parsed_content:
                        yield parsed_content
                    if is_complete:
                        pass
                else:
                    if is_first_chunk:
                        if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(
                            self.reasoning_parser
                        ):
                            text = thinking_parser.get_thinking_open() + text
                        is_first_chunk = False

                    if thinking_parser:
                        parsed_content, is_complete = thinking_parser.parse_stream(text)
                        after_thinking_close_content = None
                        if parsed_content:
                            if isinstance(parsed_content, dict):
                                after_thinking_close_content = parsed_content.pop("content", None)
                            yield parsed_content
                        if is_complete:
                            thinking_parser = None
                        if after_thinking_close_content:
                            text = after_thinking_close_content
                        else:
                            continue

                    if tool_parser:
                        parsed_content, is_complete = tool_parser.parse_stream(text)
                        if is_complete:
                            yield parsed_content
                        continue

                    yield text

            # Usage info
            completion_text = "".join(completion_chunks)
            completion_tokens = self._count_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens

            yield {
                "__usage__": UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            }

        except (asyncio.CancelledError, GeneratorExit):
            logger.info(f"Request {request_id} cancelled.")
            context.cancel()
            raise
        except asyncio.QueueFull:
            context.cancel()
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            context.cancel()
            logger.error(f"Error in text stream generation for request {request_id}: {e!s}")
            content = create_error_response(
                f"Failed to generate text stream: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)
        finally:
            context.cancel()

    async def generate_text_response(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """
        Generate a complete response for text-only chat completion requests.
        Uses the request queue for handling concurrent requests.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Returns:
            dict: Response content and usage info.
        """
        request_id = f"text-{uuid.uuid4()}"
        context = GenerationContext()

        try:
            chat_messages, model_params = await self._prepare_text_request(request)

            # Count prompt tokens
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})
            prompt_tokens = self._count_message_tokens(chat_messages, **chat_template_kwargs)

            request_data = {
                "messages": chat_messages,
                "stream": False,
                "context": context,
                **model_params,
            }
            response, prompt_tokens = await self.request_queue.submit(request_id, request_data)

            # Count completion tokens
            completion_tokens = self._count_tokens(
                response if isinstance(response, str) else response.get("content", "")
            )
            total_tokens = prompt_tokens + completion_tokens

            # Create usage info
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            # Create appropriate parsers for this model type
            thinking_parser, tool_parser = self._create_parsers()

            enable_thinking = chat_template_kwargs.get("enable_thinking", True)

            if ParserFactory.respects_enable_thinking(self.reasoning_parser):
                if not enable_thinking:
                    thinking_parser = None

            if not thinking_parser and not tool_parser:
                return {"response": response, "usage": usage}

            response_text = response

            if thinking_parser and ParserFactory.has_special_parsing(self.reasoning_parser):
                # Handle parsers with special parsing logic (e.g., harmony returns dict)
                parsed = thinking_parser.parse(response_text)
                return {"response": parsed, "usage": usage}

            if thinking_parser and ParserFactory.needs_redacted_reasoning_prefix(
                self.reasoning_parser
            ):
                # Add thinking tag to response for parsers that need it
                response_text = thinking_parser.get_thinking_open() + response_text

            parsed_response = {"reasoning_content": None, "tool_calls": None, "content": None}

            if thinking_parser:
                thinking_response, response_text = thinking_parser.parse(response_text)
                parsed_response["reasoning_content"] = thinking_response

            if tool_parser:
                tool_response, response_text = tool_parser.parse(response_text)
                parsed_response["tool_calls"] = tool_response
            parsed_response["content"] = response_text

            return {"response": parsed_response, "usage": usage}

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} cancelled.")
            context.cancel()
            raise
        except asyncio.QueueFull:
            context.cancel()
            logger.error("Too many requests. Service is at capacity.")
            content = create_error_response(
                "Too many requests. Service is at capacity.",
                "rate_limit_exceeded",
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            raise HTTPException(status_code=429, detail=content)
        except Exception as e:
            context.cancel()
            logger.error(f"Error in text response generation: {e!s}")
            content = create_error_response(
                f"Failed to generate text response: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def generate_embeddings_response(self, request: EmbeddingRequest):
        """
        Generate embeddings for a given text input.

        Args:
            request: EmbeddingRequest object containing the text input.

        Returns:
            List[float]: Embeddings for the input text.
        """
        try:
            # Create a unique request ID
            request_id = f"embeddings-{uuid.uuid4()}"
            if isinstance(request.input, str):
                request.input = [request.input]
            request_data = {"type": "embeddings", "input": request.input, "model": request.model}

            # Submit to the request queue
            response = await self.request_queue.submit(request_id, request_data)

            return response

        except Exception as e:
            logger.error(f"Error in embeddings generation: {e!s}")
            content = create_error_response(
                f"Failed to generate embeddings: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def _process_request(self, request_data: dict[str, Any]) -> tuple[Any, int] | list[float]:
        """Process a text request with KVCache reuse support.

        This is the worker function for the request queue.

        Parameters
        ----------
        request_data : dict[str, Any]
            Dictionary containing the request data.

        Returns
        -------
        tuple[Any, int] | list[float]
            For text generation: (response, prompt_tokens)
            For embeddings: list of embeddings
        """
        entry_id: int | None = None

        try:
            # Check if the request is for embeddings
            if request_data.get("type") == "embeddings":
                loop = asyncio.get_running_loop()
                return await asyncio.to_thread(self._run_model_sync, loop, request_data)

            # Extract request parameters
            messages = request_data.get("messages", [])
            stream_queue = request_data.get("stream_queue")

            # Apply message conversion if needed
            if self.converter:
                refined_messages = self.converter.convert_messages(messages)
            else:
                refined_messages = [
                    {k: v for k, v in msg.items() if v is not None} for msg in messages
                ]

            # Convert tool_calls arguments from JSON string to dict for chat template
            # Some chat templates (e.g., Qwen3) expect arguments as dict, not string
            refined_messages = self._convert_tool_calls_for_template(refined_messages)

            # Update request_data with converted messages for downstream processing
            request_data["messages"] = refined_messages

            # Re-tokenize for cache lookup
            # We must do this because we need input_tokens for cache lookup
            model_params = request_data.copy()
            chat_template_kwargs = model_params.get("chat_template_kwargs", {})
            input_tokens = self.model.tokenizer.apply_chat_template(
                refined_messages,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            # Extract token IDs from BatchEncoding if necessary
            if hasattr(input_tokens, "input_ids"):
                input_tokens = input_tokens.input_ids

            # Find matching cache
            cached_kv, prefix_len, entry_id = await self.cache_manager.find_best_match(input_tokens)

            if cached_kv:
                # Verify cache offset matches expected prefix length
                cache_offset = cached_kv[0].offset if cached_kv else 0

                if cache_offset == prefix_len:
                    # Perfect match - cache ready to use
                    logger.info(
                        f"Cache hit: reusing {prefix_len}/{len(input_tokens)} tokens "
                        f"({prefix_len / len(input_tokens) * 100:.1f}%), "
                        f"cache_offset={cache_offset}"
                    )
                elif cache_offset > prefix_len:
                    # Cache has more tokens than needed - trim to exact prefix length
                    trim_amount = cache_offset - prefix_len
                    logger.info(
                        f"Cache hit with trim: {prefix_len}/{len(input_tokens)} tokens, "
                        f"trimming {trim_amount} tokens (offset {cache_offset} -> {prefix_len})"
                    )
                    for layer_cache in cached_kv:
                        if hasattr(layer_cache, "trim"):
                            layer_cache.trim(trim_amount)
                else:
                    # cache_offset < prefix_len - shouldn't happen with our matching logic
                    logger.warning(
                        f"Cache offset too small: offset={cache_offset}, prefix_len={prefix_len}. "
                        "Creating fresh cache."
                    )
                    cached_kv = make_prompt_cache(self.model.model, self.model.max_kv_size)
                    entry_id = None
            else:
                logger.info("Cache miss: creating fresh cache")
                cached_kv = make_prompt_cache(self.model.model, self.model.max_kv_size)

            # Pass prepared data to thread
            request_data["_cached_kv"] = cached_kv
            request_data["_input_tokens"] = input_tokens
            request_data["_entry_id"] = entry_id

            # Offload to thread
            loop = asyncio.get_running_loop()
            result = await asyncio.to_thread(self._run_model_sync, loop, request_data, stream_queue)

            return result

        except Exception as e:
            import traceback

            logger.error(f"Error processing text request: {e!s}\n{traceback.format_exc()}")
            # Unlock cache entry on error
            if entry_id is not None:
                await self.cache_manager.unlock_entry(entry_id)
            gc.collect()
            raise

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the request queue and cache manager.

        Returns
        -------
        dict[str, Any]
            Dictionary with queue and cache statistics.
        """
        queue_stats = self.request_queue.get_queue_stats()
        cache_stats = await self.cache_manager.get_stats()

        return {
            "queue_stats": queue_stats,
            "cache_stats": cache_stats,
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the request queue before shutdown.

        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXLMHandler resources")
            if hasattr(self, "request_queue"):
                await self.request_queue.stop()
            if hasattr(self, "cache_manager"):
                await self.cache_manager.clear()
            logger.info("MLXLMHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXLMHandler cleanup: {e!s}")
            raise

    async def _prepare_text_request(
        self, request: ChatCompletionRequest
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        Prepare a text request by parsing model parameters and verifying the format of messages.

        Args:
            request: ChatCompletionRequest object containing the messages.

        Returns:
            Tuple containing the formatted chat messages and model parameters.
        """
        try:
            request_dict = request.model_dump()
            tools = request_dict.pop("tools", None)
            tool_choice = request_dict.pop("tool_choice", None)

            if tools:
                # Enable auto tool choice if requested via CLI flag
                if self.enable_auto_tool_choice and tool_choice == "auto":
                    request_dict["chat_template_kwargs"]["tool_choice"] = "auto"
                elif tool_choice:
                    logger.warning("Tool choice has not supported yet, will be ignored.")
                request_dict["chat_template_kwargs"]["tools"] = tools

            if request_dict.get("response_format", None):
                response_format = request_dict.pop("response_format", None)
                if response_format.get("type") == "json_schema":
                    request_dict["schema"] = response_format.get("json_schema", None).get(
                        "schema", None
                    )

            # Format chat messages and merge system messages into index 0
            chat_messages = []
            system_messages = []
            non_system_messages = []

            for message in request_dict.get("messages", []):
                # Handle content that might be a list of dictionaries (multimodal format)
                content = message.get("content", None)
                if content is None:
                    continue
                if isinstance(content, list):
                    # For LM models, extract only text content and concatenate
                    text_parts = []
                    for item in content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and item.get("text")
                        ):
                            text_parts.append(item["text"])
                    content = "\n".join(text_parts) if text_parts else ""

                message["content"] = content
                # Separate system messages from other messages
                if message.get("role") == "system":
                    system_messages.append(message)
                else:
                    non_system_messages.append(message)

            # If there are system messages, merge them into a single system message at index 0
            if system_messages:
                # Combine all system message contents
                combined_system_content = "\n\n".join(
                    [msg["content"] for msg in system_messages if msg.get("content")]
                )

                # Create merged system message using the first system message as template
                merged_system_message = system_messages[0].copy()
                merged_system_message["content"] = combined_system_content

                # Add merged system message at index 0
                chat_messages.append(merged_system_message)

            # Add all non-system messages after the merged system message
            chat_messages.extend(non_system_messages)
            return chat_messages, request_dict

        except Exception as e:
            logger.error(f"Failed to prepare text request: {e!s}")
            content = create_error_response(
                f"Failed to process request: {e!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)
