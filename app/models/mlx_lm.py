"""MLX Language Model wrapper for text generation.

This module provides a wrapper class for MLX Language Models that handles
both streaming and non-streaming inference with KVCache support.
"""

from __future__ import annotations

from collections.abc import Generator
import gc
import os
from typing import Any

from loguru import logger
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_repetition_penalty, make_sampler
from mlx_lm.utils import load
from outlines.processors import JSONLogitsProcessor

from ..core.cache_utils import resolve_cache_offset
from ..core.service_llm_engine import ServiceLLMEngine
from ..core.gpt_oss_patch import patch_gpt_oss_model
from ..utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer

DEFAULT_TEMPERATURE = os.getenv("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = os.getenv("DEFAULT_TOP_P", 0.95)
DEFAULT_TOP_K = os.getenv("DEFAULT_TOP_K", 20)
DEFAULT_MIN_P = os.getenv("DEFAULT_MIN_P", 0.0)
DEFAULT_SEED = os.getenv("DEFAULT_SEED", 0)
DEFAULT_MAX_TOKENS = os.getenv("DEFAULT_MAX_TOKENS", 8192)
DEFAULT_BATCH_SIZE = os.getenv("DEFAULT_BATCH_SIZE", 32)

# Special tokens that should be used as stop sequences
# These tokens indicate role boundaries and should stop generation
DEFAULT_STOP_TOKENS = [
    "<|observation|>",
    "<|user|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|eot_id|>",
    "<end_of_turn>",
    "<|eot|>",
]


class CachedTokenAwareRepetitionPenaltyProcessor:
    """Repetition penalty processor that considers cached token history.

    This processor is essential for proper repetition penalty calculation when
    using KV cache reuse. Without considering cached tokens, the penalty would
    only apply to newly generated tokens, leading to potential repetitions of
    words from the cached prompt.

    Based on mlx-engine's RepetitionPenaltyProcessor implementation.
    """

    def __init__(
        self,
        token_history: list[int],
        repetition_penalty: float,
        repetition_context_size: int,
    ):
        """
        Initialize processor with cached token history.

        Parameters
        ----------
        token_history : list[int]
            Previously cached/generated tokens from the prompt
        repetition_penalty : float
            Penalty factor for repeated tokens (>1.0 discourages repetition)
        repetition_context_size : int
            Number of previous tokens to consider for repetition detection
        """
        self.token_history = token_history
        self.repetition_context_size = repetition_context_size
        self.penalty_fn = make_repetition_penalty(repetition_penalty, repetition_context_size)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply repetition penalty considering both history and current tokens.

        This ensures that when cache is reused (e.g., 100 cached tokens + 20 new tokens),
        the penalty considers all 120 tokens, not just the 20 new ones.

        Parameters
        ----------
        tokens : mx.array
            Currently generated tokens in this generation step
        logits : mx.array
            Logits to apply penalty to

        Returns
        -------
        mx.array
            Modified logits with repetition penalty applied
        """
        # Calculate how many tokens we need from history to fill context window
        # For 2D tokens [batch_size, seq_len], use sequence length (axis 1)
        current_seq_len = tokens.shape[1] if tokens.ndim == 2 else len(tokens)
        num_tokens_from_history = max(self.repetition_context_size - current_seq_len, 0)

        # Get historical tokens if needed
        historical = []
        if num_tokens_from_history > 0 and self.token_history:
            historical = self.token_history[-num_tokens_from_history:]

        # Combine historical and current tokens
        if historical:
            # Convert to 2D array [1, len(historical)] to match tokens shape [1, seq_len]
            historical_mx = mx.array([historical], dtype=mx.int64)
            # Concatenate along sequence dimension (axis=1)
            all_tokens = mx.concat([historical_mx, tokens], axis=1)
        else:
            all_tokens = tokens

        # Apply penalty with full context
        return self.penalty_fn(all_tokens, logits)


class MLX_LM:
    """Wrapper class for MLX Language Model that handles both streaming and non-streaming inference.

    This class provides a unified interface for generating text responses from text prompts,
    supporting both streaming and non-streaming modes.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 32768,
        trust_remote_code: bool = False,
        chat_template_file: str | None = None,
    ):
        try:
            # Load model and tokenizer
            self.model, self.tokenizer = load(
                model_path,
                lazy=False,
                tokenizer_config={"trust_remote_code": trust_remote_code},
            )

            patch_gpt_oss_model(self.model)

            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token = self.tokenizer.bos_token
            self.model_type = self.model.model_type
            self.max_kv_size = context_length
            self.outlines_tokenizer = OutlinesTransformerTokenizer(self.tokenizer)

            # Extract and set EOS token IDs from config.json
            # Based on mlx-engine pattern: set tokenizer.eos_token_ids after load
            eos_token_ids = self._extract_eos_token_ids_from_config(model_path)
            if eos_token_ids:
                logger.info(f"Setting EOS token IDs from config: {eos_token_ids}")
                # Set eos_token_ids attribute on tokenizer
                self.tokenizer.eos_token_ids = set(eos_token_ids)
                # Also update eos_token_id to be the first one
                if self.tokenizer.eos_token_id not in eos_token_ids:
                    self.tokenizer.eos_token_id = min(eos_token_ids)
                    if hasattr(self.tokenizer, "_tokenizer"):
                        self.tokenizer._tokenizer.eos_token_id = self.tokenizer.eos_token_id

            # Get final EOS token IDs
            self.eos_token_ids = self._get_eos_token_ids_from_tokenizer()
            logger.info(f"Final EOS token IDs: {self.eos_token_ids}")
            if chat_template_file:
                if not os.path.exists(chat_template_file):
                    raise ValueError(f"Chat template file {chat_template_file} does not exist")
                with open(chat_template_file) as f:
                    self.tokenizer.chat_template = f.read()

            # Initialize the custom generation engine
            self.engine = ServiceLLMEngine(self.model, self.tokenizer)

            # Extract default stop tokens from tokenizer vocabulary
            self.default_stop_tokens = self._extract_default_stop_tokens()
            if self.default_stop_tokens:
                logger.info(f"Detected default stop tokens: {self.default_stop_tokens}")

        except Exception as e:
            raise ValueError(f"Error loading model: {e!s}")

    def _extract_eos_token_ids_from_config(self, model_path: str) -> list[int] | None:
        """Extract EOS token IDs from model's config.json.

        Based on mlx-engine pattern: read config.json directly to get eos_token_id.

        Some models have multiple EOS tokens defined in their config.
        For example, GLM-4 has [151329, 151336, 151338] for
        <|endoftext|>, <|user|>, and <|observation|>.

        Parameters
        ----------
        model_path : str
            Path to the model directory containing config.json

        Returns
        -------
        list[int] | None
            List of EOS token IDs, or None if not found in config
        """
        try:
            import json
            from pathlib import Path

            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                logger.debug(f"config.json not found at {config_path}")
                return None

            with open(config_path) as f:
                config = json.load(f)

            # Check root level first, then text_config (for multimodal models)
            eos_token_id = config.get("eos_token_id")
            if eos_token_id is None:
                eos_token_id = config.get("text_config", {}).get("eos_token_id")

            if eos_token_id is not None:
                # Convert to list if it's an int
                if isinstance(eos_token_id, int):
                    return [eos_token_id]
                if isinstance(eos_token_id, list):
                    return list(set(eos_token_id))  # Remove duplicates

        except Exception as e:
            logger.debug(f"Failed to read eos_token_id from config.json: {e}")

        return None

    def _get_eos_token_ids_from_tokenizer(self) -> list[int]:
        """Get EOS token IDs from loaded tokenizer.

        After tokenizer is loaded with eos_token_ids parameter,
        the tokenizer should have eos_token_ids attribute.

        Returns
        -------
        list[int]
            List of EOS token IDs from tokenizer
        """
        # Check if tokenizer has eos_token_ids attribute (set by mlx_lm.load)
        if hasattr(self.tokenizer, "eos_token_ids"):
            eos_ids = self.tokenizer.eos_token_ids
            if isinstance(eos_ids, set):
                return list(eos_ids)
            if isinstance(eos_ids, list):
                return eos_ids
            if isinstance(eos_ids, int):
                return [eos_ids]

        # Fallback to single eos_token_id
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            return [self.tokenizer.eos_token_id]

        logger.warning("No EOS token IDs found in tokenizer")
        return []

    def _extract_default_stop_tokens(self) -> list[str]:
        """Extract default stop tokens from the tokenizer vocabulary.

        Note: This extracts role/control tokens that should stop generation,
        excluding EOS tokens which are handled separately via eos_token_ids.

        Returns
        -------
        list[str]
            List of stop tokens that exist in the tokenizer's vocabulary.
        """
        stop_tokens = []
        try:
            vocab = self.tokenizer._tokenizer.get_vocab()

            # Get EOS token strings to exclude them from stop sequences
            # (they're handled separately as eos_token_ids)
            eos_token_strings = set()
            for eos_id in self.eos_token_ids:
                try:
                    eos_str = self.tokenizer.decode([eos_id])
                    eos_token_strings.add(eos_str)
                except Exception:
                    pass

            for token in DEFAULT_STOP_TOKENS:
                if token in vocab and token not in eos_token_strings:
                    stop_tokens.append(token)
                    logger.debug(f"Added stop token: '{token}'")
                elif token in eos_token_strings:
                    logger.debug(f"Skipping '{token}' (handled as EOS token)")

        except Exception as e:
            logger.warning(f"Failed to extract stop tokens from vocabulary: {e}")
        return stop_tokens

    def _apply_pooling_strategy(self, embeddings: mx.array) -> mx.array:
        embeddings = mx.mean(embeddings, axis=1)
        return embeddings

    def _apply_l2_normalization(self, embeddings: mx.array) -> mx.array:
        l2_norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (l2_norms + 1e-8)
        return embeddings

    def _batch_process(
        self, prompts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[int]]:
        """Process prompts in batches with optimized tokenization."""
        all_tokenized = []

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tokenized_batch = []

            # Tokenize all prompts in batch
            for p in batch:
                add_special_tokens = self.bos_token is None or not p.startswith(self.bos_token)
                tokens = self.tokenizer.encode(p, add_special_tokens=add_special_tokens)
                tokenized_batch.append(tokens)

            # Find max length in batch
            max_length = max(len(tokens) for tokens in tokenized_batch)

            # Pad tokens in a vectorized way
            for tokens in tokenized_batch:
                padding = [self.pad_token_id] * (max_length - len(tokens))
                all_tokenized.append(tokens + padding)

        return all_tokenized

    def _preprocess_prompt(self, prompt: str) -> list[int]:
        """Tokenize a single prompt efficiently."""
        add_special_tokens = self.bos_token is None or not prompt.startswith(self.bos_token)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return mx.array(tokens)

    def get_model_type(self) -> str:
        return self.model_type

    def get_embeddings(
        self,
        prompts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ) -> list[float]:
        """Get embeddings for a list of prompts efficiently.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts.
        batch_size : int
            Size of batches for processing.
        normalize : bool
            Whether to apply L2 normalization.

        Returns
        -------
        list[float]
            List of embeddings as float arrays.
        """
        # Process in batches to optimize memory usage
        all_embeddings = []
        try:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                tokenized_batch = self._batch_process(batch_prompts, batch_size)

                # Convert to MLX array for efficient computation
                tokenized_batch = mx.array(tokenized_batch)

                try:
                    # Compute embeddings for batch
                    batch_embeddings = self.model.model(tokenized_batch)
                    pooled_embedding = self._apply_pooling_strategy(batch_embeddings)
                    if normalize:
                        pooled_embedding = self._apply_l2_normalization(pooled_embedding)
                    all_embeddings.extend(pooled_embedding.tolist())
                finally:
                    # Explicitly free MLX arrays to prevent memory leaks
                    del tokenized_batch
                    if "batch_embeddings" in locals():
                        del batch_embeddings
                    if "pooled_embedding" in locals():
                        del pooled_embedding
                    # Force MLX garbage collection
                    mx.clear_cache()
                    gc.collect()
        except Exception:
            # Clean up on error
            mx.clear_cache()
            gc.collect()
            raise

        return all_embeddings

    def __call__(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        prompt_cache: Any | None = None,
        context: Any | None = None,
        **kwargs,
    ) -> tuple[str | Generator[str, None, None], int, Any]:
        """Generate text response from the model.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of messages in the conversation.
        stream : bool
            Whether to stream the response.
        prompt_cache : Any | None
            Optional pre-filled KVCache for reuse. If None, a new cache is created.
        context : Any | None
            Optional GenerationContext for cancellation.
        **kwargs
            Additional parameters for generation:
            - temperature: Sampling temperature (default: 0.7)
            - top_p: Top-p sampling parameter (default: 0.95)
            - top_k: Top-k sampling parameter (default: 20)
            - min_p: Min-p sampling parameter (default: 0.0)
            - seed: Random seed (default: 0)
            - max_tokens: Maximum tokens to generate (default: 8192)
            - cache_offset: Explicit cache offset override (default: None)

        Returns
        -------
        tuple[str | Generator[str, None, None], int, Any]
            A tuple of (response, prompt_tokens, cache):
            - Non-streaming: (response_text, prompt_tokens, cache)
            - Streaming: (response_generator, prompt_tokens, cache)
        """
        # Set default parameters if not provided
        seed = kwargs.get("seed", DEFAULT_SEED)
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        else:
            max_tokens = int(max_tokens)

        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})

        # Merge user-provided stop sequences with model's default stop tokens
        user_stop_sequences = kwargs.get("stop") or []
        if isinstance(user_stop_sequences, str):
            user_stop_sequences = [user_stop_sequences]

        # Combine default stop tokens with user-provided ones (avoid duplicates)
        stop_sequences = list(self.default_stop_tokens)
        for seq in user_stop_sequences:
            if seq not in stop_sequences:
                stop_sequences.append(seq)

        sampler_kwargs = {
            "temp": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": kwargs.get("top_p", DEFAULT_TOP_P),
            "top_k": kwargs.get("top_k", DEFAULT_TOP_K),
            "min_p": kwargs.get("min_p", DEFAULT_MIN_P),
        }

        repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        repetition_context_size = kwargs.get("repetition_context_size", 20)

        mx.random.seed(seed)

        # Use provided cache or create new one
        if prompt_cache is None:
            prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )

        # Extract token IDs from BatchEncoding if necessary
        if hasattr(input_tokens, "input_ids"):
            input_tokens = input_tokens.input_ids

        # Extract cache offset to determine cached tokens for repetition penalty
        cache_offset = resolve_cache_offset(prompt_cache, kwargs.get("cache_offset"))

        # Calculate cached tokens for proper repetition penalty
        # This ensures that when cache is reused, repetition penalty considers
        # both cached prompt tokens and newly generated tokens
        cached_tokens = []
        if cache_offset > 0:
            # Only include tokens that are actually in the cache
            cached_tokens = input_tokens[:cache_offset]
            logger.debug(f"Repetition penalty will consider {len(cached_tokens)} cached tokens")

        # Build logits processors with cache-aware repetition penalty
        logits_processors = []

        if repetition_penalty != 1.0:
            # Use cache-aware processor instead of standard one
            logits_processors.append(
                CachedTokenAwareRepetitionPenaltyProcessor(
                    token_history=cached_tokens,
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=repetition_context_size,
                )
            )

        json_schema = kwargs.get("schema")
        if json_schema:
            logits_processors.append(
                JSONLogitsProcessor(
                    schema=json_schema,
                    tokenizer=self.outlines_tokenizer,
                    tensor_library_name="mlx",
                )
            )

        sampler = make_sampler(**sampler_kwargs)
        prompt_tokens_len = len(input_tokens)

        # Skip already-processed tokens based on cache offset
        # This enables efficient prefix reuse - only process new tokens
        # Note: cache_offset already calculated above for repetition penalty

        # Convert to MLX array
        input_tokens_mx = mx.array(input_tokens)

        if cache_offset > 0:
            if cache_offset < len(input_tokens):
                # Cache has partial prefix - only process remaining tokens
                logger.info(f"Cache reuse: skipping {cache_offset}/{len(input_tokens)} tokens")
                input_tokens_mx = input_tokens_mx[cache_offset:]
            else:
                # cache_offset >= len(input_tokens)
                # Cache has all prompt tokens - just need last one to trigger generation
                logger.info(
                    f"Cache full hit: {cache_offset} >= {len(input_tokens)} tokens, "
                    "using last token only"
                )
                # Keep last token - needs at least 1 token
                input_tokens_mx = input_tokens_mx[-1:]

        # Use ServiceLLMEngine for generation
        response_gen = self.engine.generate_stream(
            prompt_tokens=input_tokens_mx,
            prompt_cache=prompt_cache,
            context=context,
            max_tokens=max_tokens,
            sampler=sampler,
            stop_sequences=stop_sequences,
            logits_processors=logits_processors,
            eos_token_ids=self.eos_token_ids,
        )

        if not stream:
            # Non-streaming: consume generator
            response_text = ""
            for chunk in response_gen:
                response_text += chunk
            return response_text, prompt_tokens_len, prompt_cache
        # Streaming mode: return generator of chunks
        return response_gen, prompt_tokens_len, prompt_cache
