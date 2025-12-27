from collections.abc import Generator
import threading
from typing import Any

from loguru import logger
import mlx.core as mx

from app.core.gpt_oss_patch import patch_gpt_oss_model


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
    """Handles stop sequences for generation with partial match buffering."""

    def __init__(self, stop_sequences: list[str] = None, tokenizer: Any = None):
        self.stop_sequences = stop_sequences or []
        self.tokenizer = tokenizer
        self.current_text = ""

    def process(self, new_text: str) -> tuple[str, bool]:
        """
        Process new text chunk and check for stop sequences.
        Returns (text_chunk_to_yield, should_stop).
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
        # We want to keep the longest suffix that IS a prefix of some stop_sequence
        longest_partial_len = 0

        for stop_seq in self.stop_sequences:
            # Check all possible suffix lengths
            # Optimization: only check up to len(current_text) and len(stop_seq)
            max_check_len = min(len(self.current_text), len(stop_seq))
            for length in range(max_check_len, 0, -1):
                suffix = self.current_text[-length:]
                if stop_seq.startswith(suffix):
                    longest_partial_len = max(longest_partial_len, length)
                    # Found a match for this length, no need to check shorter for this stop_seq
                    break

        if longest_partial_len > 0:
            # We have a partial match.
            # Yield the safe part, keep the partial match in buffer.
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

        # Apply gpt-oss patch for mixed attention models
        # This fixes mask broadcasting issues with chunked prefill
        self._is_patched_gpt_oss = patch_gpt_oss_model(model)

    def generate_stream(
        self,
        prompt_tokens: mx.array,
        prompt_cache: Any,
        context: GenerationContext | None = None,
        max_tokens: int = 100,
        sampler: Any = None,
        stop_sequences: list[str] = None,
        logits_processors: list[Any] = None,
    ) -> Generator[str, None, None]:
        """
        Stream generation with explicit cancellation support and chunked prefill.
        """
        if context is None:
            context = GenerationContext()
        if logits_processors is None:
            logits_processors = []

        stop_handler = StopSequenceHandler(stop_sequences, self.tokenizer)
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

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

        # Chunked prefill for all models (gpt-oss is patched in __init__)
        for i in range(0, N - 1, batch_size):
            if context.is_cancelled:
                logger.info("Request cancelled during prefill.")
                return

            chunk = prompt_tokens[:, i : min(i + batch_size, N - 1)]

            # Forward pass updates cache in-place
            logits = self.model(chunk, cache=prompt_cache)
            mx.eval(logits)

        # 2. Prepare for Decode Loop
        curr_token = prompt_tokens[:, -1:]

        # 3. Decode Loop
        try:
            for _ in range(max_tokens):
                # [CRITICAL] Check cancellation before GPU op
                if context.is_cancelled:
                    logger.info("Request cancelled during decode.")
                    break

                # Inference
                logits = self.model(curr_token, cache=prompt_cache)

                # Apply logits processors
                logits = logits[:, -1, :]
                for processor in logits_processors:
                    logits = processor(curr_token, logits)

                # Sampling
                if sampler:
                    next_token = sampler(logits)
                else:
                    next_token = mx.argmax(logits, axis=-1)

                curr_token = next_token[:, None]
                mx.eval(curr_token)

                # Decode and Check Stop
                token_item = next_token.item()

                # Use StreamingDetokenizer for proper unicode handling
                detokenizer.add_token(token_item)
                new_text = detokenizer.last_segment

                # StopSequenceHandler now expects text string
                text_chunk, stopped = stop_handler.process(new_text)

                if text_chunk:
                    yield text_chunk

                if stopped or token_item == self.tokenizer.eos_token_id:
                    break

        finally:
            # Deterministic Cleanup if needed
            # In a shared service, we might NOT want to clear cache if we want to reuse it.
            # But we should ensure we don't leave hanging graph nodes.
            mx.clear_cache()

    def sample(self, logits):
        return mx.argmax(logits, axis=-1)
