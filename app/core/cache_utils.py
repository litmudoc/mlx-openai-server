"""Utilities for inspecting and using prompt caches."""

from __future__ import annotations

from typing import Any


def cache_has_offset(prompt_cache: list[Any] | None) -> bool:
    """Check whether a prompt cache exposes an offset attribute.

    Parameters
    ----------
    prompt_cache : list[Any] | None
        Cache list from mlx_lm or None.

    Returns
    -------
    bool
        True when an offset attribute is available.
    """
    if not prompt_cache:
        return False
    return hasattr(prompt_cache[0], "offset")


def resolve_cache_offset(
    prompt_cache: list[Any] | None, provided_offset: int | None = None
) -> int:
    """Resolve cache offset from explicit value or cache metadata.

    Parameters
    ----------
    prompt_cache : list[Any] | None
        Cache list from mlx_lm or None.
    provided_offset : int | None
        Explicit offset (e.g., prefix length from cache matching).

    Returns
    -------
    int
        Resolved cache offset, or 0 if unavailable.
    """
    if provided_offset is not None:
        return int(provided_offset)
    if not prompt_cache:
        return 0
    offset = getattr(prompt_cache[0], "offset", None)
    if offset is None:
        return 0
    return int(offset)
