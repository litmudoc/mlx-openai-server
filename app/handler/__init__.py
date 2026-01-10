"""
MLX model handlers for text, multimodal, image generation, and embeddings models.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MLXLMHandler",
    "MLXVLMHandler",
    "MLXFluxHandler",
    "MLXEmbeddingsHandler",
    "MFLUX_AVAILABLE",
]

_MFLUX_AVAILABLE: bool | None = None


def __getattr__(name: str) -> Any:
    """Lazy-load handlers to avoid importing MLX at package import time."""
    global _MFLUX_AVAILABLE
    if name == "MLXLMHandler":
        from .mlx_lm import MLXLMHandler

        return MLXLMHandler
    if name == "MLXVLMHandler":
        from .mlx_vlm import MLXVLMHandler

        return MLXVLMHandler
    if name == "MLXEmbeddingsHandler":
        from .mlx_embeddings import MLXEmbeddingsHandler

        return MLXEmbeddingsHandler
    if name == "MLXFluxHandler":
        try:
            from .mflux import MLXFluxHandler

            _MFLUX_AVAILABLE = True
            return MLXFluxHandler
        except ImportError:
            _MFLUX_AVAILABLE = False
            raise
    if name == "MFLUX_AVAILABLE":
        if _MFLUX_AVAILABLE is None:
            try:
                from .mflux import MLXFluxHandler as _

                _MFLUX_AVAILABLE = True
            except ImportError:
                _MFLUX_AVAILABLE = False
        return _MFLUX_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
