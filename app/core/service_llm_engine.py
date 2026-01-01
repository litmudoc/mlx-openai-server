from collections.abc import Callable, Generator
from dataclasses import dataclass
import threading
from typing import Any

from loguru import logger
import mlx.core as mx

# Generation-dedicated stream for optimized async GPU execution
# Based on mlx-lm pattern for better GPU utilization and pipelining
# See analysis/prefill-decode-optimization.md for details
generation_stream = mx.new_stream(mx.default_device())


@dataclass
class Token:
    """Represents a generated token with metadata.

    Based on mlx-engine's Token structure for detailed generation tracking.
    """

    token_id: int
    text: str
    logprob: float = 0.0


@dataclass
class GenerationResult:
    """Structured generation result with detailed metadata.

    Provides comprehensive information about generation including:
    - Generated text
    - Token-level details (IDs, text, probabilities)
    - Stop condition information

    Based on mlx-engine's GenerationResult pattern.
    """

    text: str
    tokens: list[Token]
    stop_condition: str | None = None
    stop_tokens: list[int] | None = None


class GenerationContext:
    """Controls the lifecycle of a generation request."""

    def __init__(self):
        self._cancel_flag = threading.Event()

    def cancel(self):
        self._cancel_flag.set()

    @property
    def is_cancelled(self):
        return self._cancel_flag.is_set()


class StopSequenceHandler:
    """Handles stop sequences for generation with partial match buffering.

    Enhanced version based on mlx-engine's StopStringProcessor with:
    - Token ID buffer for debugging and stop_tokens tracking
    - UTF-8 incomplete character detection
    - Earliest match selection for multiple stop sequences
    """

    def __init__(self, stop_sequences: list[str] = None, tokenizer: Any = None):
        self.stop_sequences = stop_sequences or []
        self.tokenizer = tokenizer
        self.current_text = ""
        self.token_buffer = []  # Track token IDs for debugging and stop_tokens

    def process_token(self, token: int) -> tuple[str, bool, list[int] | None]:
        """
        Process new token and check for stop sequences.

        Parameters
        ----------
        token : int
            Token ID to process

        Returns
        -------
        tuple[str, bool, list[int] | None]
            - text_chunk_to_yield: Text safe to yield
            - should_stop: Whether to stop generation
            - stop_tokens: List of token IDs where stop string was found (if stopped)
        """
        self.token_buffer.append(token)
        decoded = self.tokenizer.decode(self.token_buffer)

        # 1. Check for incomplete UTF-8 (multi-byte character)
        # The replacement character indicates incomplete UTF-8 sequence
        if decoded and decoded[-1] == "\ufffd":
            return "", False, None

        # 2. Check for full match (earliest match among all stop sequences)
        earliest_match = {"position": float("inf"), "stop_string": None}

        for stop_seq in self.stop_sequences:
            position = decoded.find(stop_seq)
            if position != -1 and position < earliest_match["position"]:
                earliest_match["position"] = position
                earliest_match["stop_string"] = stop_seq

        if earliest_match["stop_string"] is not None:
            from loguru import logger

            logger.info(
                f"Stop sequence matched! "
                f"stop_string='{earliest_match['stop_string']}', "
                f"position={earliest_match['position']}, "
                f"decoded_buffer='{decoded}', "
                f"token_buffer={self.token_buffer}"
            )
            final_text = decoded[: earliest_match["position"]]
            stop_tokens = self.token_buffer.copy()
            self.token_buffer.clear()
            return final_text, True, stop_tokens

        # 3. Check for partial match to buffer
        # We want to keep the longest suffix that IS a prefix of some stop_sequence
        longest_partial_len = 0

        for stop_seq in self.stop_sequences:
            # Check all possible suffix lengths
            max_check_len = min(len(decoded), len(stop_seq))
            for length in range(max_check_len, 0, -1):
                suffix = decoded[-length:]
                if stop_seq.startswith(suffix):
                    longest_partial_len = max(longest_partial_len, length)
                    # Found a match for this length, no need to check shorter
                    break

        if longest_partial_len > 0:
            # We have a partial match.
            # Yield the safe part, keep buffer for next iteration
            safe_part = decoded[:-longest_partial_len]
            # Don't clear buffer - keep for potential match
            return safe_part, False, None

        # No partial match, yield everything and clear buffer
        chunk = decoded
        self.token_buffer.clear()
        return chunk, False, None

    def process(self, new_text: str) -> tuple[str, bool]:
        """
        Legacy method for backward compatibility.
        Process new text chunk and check for stop sequences.
        Returns (text_chunk_to_yield, should_stop).

        Note: This method doesn't have access to token IDs,
        so it cannot provide stop_tokens or UTF-8 detection.
        Use process_token() for enhanced functionality.
        """
        self.current_text += new_text

        # 1. Check for full match
        for stop_seq in self.stop_sequences:
            if stop_seq in self.current_text:
                stop_index = self.current_text.find(stop_seq)
                final_chunk = self.current_text[:stop_index]
                self.current_text = ""  # Clear buffer
                return final_chunk, True

        # 2. Check for partial match to buffer
        longest_partial_len = 0

        for stop_seq in self.stop_sequences:
            max_check_len = min(len(self.current_text), len(stop_seq))
            for length in range(max_check_len, 0, -1):
                suffix = self.current_text[-length:]
                if stop_seq.startswith(suffix):
                    longest_partial_len = max(longest_partial_len, length)
                    break

        if longest_partial_len > 0:
            safe_part = self.current_text[:-longest_partial_len]
            self.current_text = self.current_text[-longest_partial_len:]
            return safe_part, False

        # No partial match, yield everything
        chunk = self.current_text
        self.current_text = ""
        return chunk, False


class ServiceLLMEngine:
    """
    Core LLM generation engine with support for:
    1. Chunked Prefill (Memory Management)
    2. Explicit Interruption/Cancellation
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_stream(
        self,
        prompt_tokens: mx.array,
        prompt_cache: Any,
        context: GenerationContext | None = None,
        max_tokens: int = 100,
        sampler: Any = None,
        stop_sequences: list[str] = None,
        logits_processors: list[Any] = None,
        prompt_progress_callback: Callable[[float], bool] | None = None,
        eos_token_ids: list[int] | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream generation with explicit cancellation support and chunked prefill.

        Parameters
        ----------
        prompt_tokens : mx.array
            Input tokens to process
        prompt_cache : Any
            KV cache for the model
        context : GenerationContext | None
            Context for cancellation control
        max_tokens : int
            Maximum tokens to generate
        sampler : Any
            Sampling function
        stop_sequences : list[str] | None
            Stop sequences for generation
        logits_processors : list[Any] | None
            Logits processors to apply
        prompt_progress_callback : Callable[[float], bool] | None
            Optional callback for prefill progress reporting.
            Receives progress as float (0-100).
            Should return True to continue or False to cancel.
            Based on mlx-engine's progress callback pattern.
        """
        if context is None:
            context = GenerationContext()
        if logits_processors is None:
            logits_processors = []
        if eos_token_ids is None:
            # Fallback to tokenizer's single EOS token
            eos_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []

        stop_handler = StopSequenceHandler(stop_sequences, self.tokenizer)

        # Log configuration for debugging
        logger.info(f"Stop sequences configured: {stop_sequences}")
        logger.info(f"EOS token IDs configured: {eos_token_ids}")

        # Token buffer for detailed generation tracking
        # Based on mlx-engine pattern for better debugging and analysis
        token_buffer: list[Token] = []
        accumulated_text = ""

        # 1. Chunked Prefill (Memory Safe)
        # We assume prompt_cache matches the beginning of prompt_tokens if partially filled
        # Logic to skip already cached tokens should be handled before calling this,
        # OR we can handle it here if we pass the full prompt and check cache offset.

        # Current logic in MLX_LM prepares input_tokens.
        # If we receive input_tokens that only contain the *new* part (suffix),
        # we just process them.
        # But for Chunked Prefill, we need to handle the case where input_tokens is large.

        # Chunked prefill configuration.
        # With gpt-oss patch applied, all models can use chunked prefill.
        # The patch fixes mask broadcasting issues for mixed attention models.
        # See analysis/gpt_oss_chunked_prefill_issue.md for details.
        batch_size = 512  # Default prefill batch size

        # input_tokens is expected to be [1, N] or [N]
        if len(prompt_tokens.shape) == 1:
            prompt_tokens = prompt_tokens[None, :]

        N = prompt_tokens.shape[1]

        # Use generation stream for optimized GPU execution
        # Based on mlx-lm pattern for better performance and pipelining
        with mx.stream(generation_stream):
            # Chunked prefill for all models (gpt-oss is patched in __init__)
            for i in range(0, N - 1, batch_size):
                if context.is_cancelled:
                    logger.info("Request cancelled during prefill.")
                    return

                chunk = prompt_tokens[:, i : min(i + batch_size, N - 1)]

                # Forward pass updates cache in-place
                logits = self.model(chunk, cache=prompt_cache)

                # Evaluate cache state explicitly (critical for cache consistency)
                # Based on mlx-engine pattern
                mx.eval([c.state for c in prompt_cache])
                mx.eval(logits)
                mx.clear_cache()

                # Report prefill progress if callback provided
                if prompt_progress_callback:
                    # Calculate progress as percentage of tokens processed
                    progress = min(100.0, ((i + batch_size) / (N - 1)) * 100)
                    should_continue = prompt_progress_callback(progress)

                    # If callback returns False, cancel generation
                    if should_continue is False:
                        logger.info("Prefill cancelled by progress callback.")
                        context.cancel()
                        return

            # 2. Prepare for Decode Loop - First token generation
            # Generate first token within prefill block for better pipelining
            curr_token = prompt_tokens[:, -1:]
            logits = self.model(curr_token, cache=prompt_cache)
            logits = logits[:, -1, :]

            # Apply logits processors for first token
            for processor in logits_processors:
                logits = processor(curr_token, logits)

            mx.eval(logits)

            # Sample first token
            if sampler:
                first_token = sampler(logits)
            else:
                first_token = mx.argmax(logits, axis=-1)

        # Async eval for first token (enables prefetching)
        mx.async_eval(first_token)

        # Helper function for single decode step with stream
        # Based on mlx-lm pattern for prefetching and pipelining
        def _step(token: mx.array) -> mx.array:
            """Generate next token using generation stream."""
            with mx.stream(generation_stream):
                # Ensure token is 2D [batch_size, 1] for consistency
                token_2d = token[:, None] if token.ndim == 1 else token
                logits = self.model(token_2d, cache=prompt_cache)
                logits = logits[:, -1, :]

                # Apply logits processors with 2D token for shape consistency
                for processor in logits_processors:
                    logits = processor(token_2d, logits)

                # Sample next token
                if sampler:
                    next_token = sampler(logits)
                else:
                    next_token = mx.argmax(logits, axis=-1)

                return next_token

        # 3. Decode Loop with Prefetching
        # Based on mlx-lm pattern: compute next token while processing current token
        try:
            # Synchronize first token
            mx.eval(first_token)
            current_token = first_token

            # Log max_tokens for debugging
            logger.info(f"Starting decode loop: max_tokens={max_tokens}")

            for step in range(max_tokens):
                # [CRITICAL] Check cancellation before GPU op
                if context.is_cancelled:
                    logger.info("Request cancelled during decode.")
                    break

                # Prefetch next token asynchronously (pipelining)
                if step < max_tokens - 1:
                    next_token = _step(current_token)
                    mx.async_eval(next_token)

                # Process current token (while next token is computing)
                token_item = int(current_token.item())

                # [MEMORY] Clear cache after extracting token value
                mx.clear_cache()

                # Check EOS token FIRST before decoding
                # This prevents EOS token from being output
                if token_item in eos_token_ids:
                    # Decode to show which EOS token was encountered
                    eos_decoded = self.tokenizer.decode([token_item])
                    logger.info(
                        f"Generation stopped by EOS token. "
                        f"EOS token ID: {token_item}, "
                        f"Decoded: '{eos_decoded}', "
                        f"Total tokens generated: {len(token_buffer)}"
                    )
                    break

                # Decode and Check Stop Sequences
                # Use enhanced StopSequenceHandler with token-based processing
                text_chunk, stopped, stop_tokens = stop_handler.process_token(token_item)

                # Track token metadata for debugging and analysis
                # TODO: Extract actual logprob from logits if needed for detailed analysis
                if text_chunk:
                    accumulated_text += text_chunk
                    token_buffer.append(
                        Token(
                            token_id=token_item,
                            text=text_chunk,
                            logprob=0.0,  # Future: extract from logits
                        )
                    )

                if text_chunk:
                    yield text_chunk

                if stopped:
                    if stop_tokens:
                        # Decode stop tokens to see what was matched
                        decoded_stop = self.tokenizer.decode(stop_tokens)
                        logger.info(
                            f"Generation stopped by stop sequence. "
                            f"Stop tokens: {stop_tokens}, "
                            f"Decoded: '{decoded_stop}', "
                            f"Total tokens generated: {len(token_buffer)}"
                        )
                    break

                # Move to next token (already computed if step < max_tokens - 1)
                if step < max_tokens - 1:
                    # Next token is already being computed, just synchronize
                    mx.eval(next_token)
                    current_token = next_token

        finally:
            # Log generation statistics for debugging and monitoring
            if token_buffer:
                # Determine stop reason
                stop_reason = "unknown"
                if context.is_cancelled:
                    stop_reason = "cancelled"
                elif step >= max_tokens - 1:
                    stop_reason = f"max_tokens_reached (max_tokens={max_tokens})"
                elif token_item in eos_token_ids:
                    stop_reason = f"eos_token (ID: {token_item})"
                elif stopped:
                    stop_reason = "stop_sequence"

                logger.info(
                    f"Generation completed: {len(token_buffer)} tokens generated, "
                    f"{len(accumulated_text)} characters, "
                    f"stop_reason={stop_reason}"
                )

            # Deterministic Cleanup if needed
            # In a shared service, we might NOT want to clear cache if we want to reuse it.
            # But we should ensure we don't leave hanging graph nodes.
            mx.clear_cache()

    def sample(self, logits):
        return mx.argmax(logits, axis=-1)
