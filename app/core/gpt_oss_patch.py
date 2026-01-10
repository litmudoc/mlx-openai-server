"""Monkey patch for gpt-oss model to fix chunked prefill mask broadcasting issues.

The gpt-oss model uses a mixed attention architecture with both full attention
(KVCache) and sliding window attention (RotatingKVCache) layers. The original
implementation generates masks once for all layers, but this causes broadcasting
errors when:

1. Using chunked prefill with batch_size > sliding_window
2. Processing long sequences where create_causal_mask produces incompatible shapes

This patch modifies the model's __call__ method to generate masks per-layer,
ensuring each layer uses a mask compatible with its specific cache state.

See analysis/gpt_oss_chunked_prefill_issue.md for detailed analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
import mlx.core as mx

if TYPE_CHECKING:
    from mlx_lm.models.gpt_oss import Model

_PATCHED_MODELS: set[int] = set()


def _make_patched_call():
    """Create a patched __call__ function for GptOssMoeModel.

    Returns
    -------
    callable
        The patched __call__ function.
    """
    # Import here to avoid issues at module load time
    from mlx_lm.models.base import create_attention_mask

    def patched_call(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        """Patched __call__ that generates masks per-layer.

        This fixes the mask broadcasting issue by generating attention masks
        for each layer individually based on its specific cache state.

        Key fix: For sliding window layers, we generate an explicit mask that:
        1. Respects the physical shape of RotatingKVCache (avoiding broadcast errors)
        2. Uses global offsets to maintain correct context visibility
        3. Properly handles the sliding window constraint
        """
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Key fix: Generate masks per-layer with correct strategy
        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            if layer_type == "full_attention":
                # Full attention: use standard mask generation
                mask = create_attention_mask(x, c)
            else:
                # Sliding window attention
                N = x.shape[1]
                if N == 1:
                    # Single token decode: no mask needed
                    mask = None
                else:
                    # Prefill: Generate explicit mask based on global positions
                    # See analysis/gpt_oss_chunked_prefill_issue.md
                    
                    # Determine offset and window size
                    if c is not None:
                        current_offset = c.offset
                        window_size = c.max_size
                    else:
                        current_offset = 0
                        window_size = self.window_size

                    # Calculate L: size of keys after update
                    # Keys will be truncated to window_size if they exceed it
                    L = min(current_offset + N, window_size)

                    # Generate indices
                    # Queries: [current_offset, ..., current_offset + N - 1]
                    queries_pos = mx.arange(N) + current_offset
                    
                    # Keys: [current_offset + N - L, ..., current_offset + N - 1]
                    # This maps physical indices 0..L-1 to global positions
                    keys_pos = mx.arange(L) + (current_offset + N - L)
                    
                    # Broadcast for comparison
                    queries_pos = queries_pos[:, None]
                    keys_pos = keys_pos[None, :]
                    
                    # Mask conditions:
                    # 1. Causal: keys must be in the past or present relative to query
                    # 2. Window: keys must be within window_size of query
                    mask_bool = (keys_pos <= queries_pos) & (keys_pos > (queries_pos - window_size))
                    
                    # Create additive mask (-inf for masked, 0 for allowed)
                    mask = mx.where(mask_bool, 0, -float("inf"))
                    mask = mask.astype(x.dtype)

            x = layer(x, mask, c)

        return self.norm(x)

    return patched_call


def patch_gpt_oss_model(model: Model) -> bool:
    """Apply monkey patch to gpt-oss model if needed.

    Parameters
    ----------
    model : Model
        The gpt-oss Model instance to patch.

    Returns
    -------
    bool
        True if patch was applied, False if already patched or not applicable.
    """
    # Check if this model instance is already patched
    model_id = id(model)
    if model_id in _PATCHED_MODELS:
        logger.debug("gpt-oss model already patched, skipping")
        return False

    # Verify this is a gpt-oss model
    if not hasattr(model, "model"):
        logger.debug("Model has no 'model' attribute, not a gpt-oss model")
        return False

    inner_model = model.model
    model_class_name = inner_model.__class__.__name__

    if model_class_name != "GptOssMoeModel":
        logger.debug(f"Inner model is {model_class_name}, not GptOssMoeModel")
        return False

    # Verify it has the expected attributes
    required_attrs = ["layers", "layer_types", "window_size", "embed_tokens", "norm"]
    if not all(hasattr(inner_model, attr) for attr in required_attrs):
        missing = [attr for attr in required_attrs if not hasattr(inner_model, attr)]
        logger.warning(f"gpt-oss model missing attributes: {missing}, skipping patch")
        return False

    # Apply the patch by replacing the __call__ method on the class
    # This is more reliable than instance method replacement
    inner_model_class = inner_model.__class__

    # Store original method if not already stored
    if not hasattr(inner_model_class, "_original_call"):
        inner_model_class._original_call = inner_model_class.__call__
        inner_model_class.__call__ = _make_patched_call()
        logger.info("Applied gpt-oss chunked prefill patch (class-level) for mixed attention")

    _PATCHED_MODELS.add(model_id)
    return True


def is_gpt_oss_model(model: Model) -> bool:
    """Check if a model is a gpt-oss model.

    Parameters
    ----------
    model : Model
        The model to check.

    Returns
    -------
    bool
        True if the model is a gpt-oss model.
    """
    if not hasattr(model, "model"):
        return False
    
    class_name = model.model.__class__.__name__
    return "gptoss" in class_name.lower() or class_name == "GptOssMoeModel"
