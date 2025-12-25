"""Patch mlx_lm cache classes for proper offset tracking.

This module patches RotatingKVCache to properly track the offset after
update_and_fetch, enabling efficient cache reuse for prefix matching.

Based on mlx_textgen's implementation.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

_PATCHED = False


def _new_update_and_fetch(self: RotatingKVCache, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
    """Update cache and fetch state with proper offset tracking.

    Parameters
    ----------
    k : mx.array
        Key tensor to add to cache.
    v : mx.array
        Value tensor to add to cache.

    Returns
    -------
    tuple[mx.array, mx.array]
        The current cache state (keys, values).
    """
    KVCache.update_and_fetch(self, k, v)
    self._idx = self.keys.shape[2]
    return self.state


def _new_state_getter(self: RotatingKVCache) -> tuple[mx.array, mx.array]:
    """Get the current cache state with proper offset handling.

    Returns
    -------
    tuple[mx.array, mx.array]
        The current cache state (keys, values).
    """
    if self.offset <= self.max_size:
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
    elif self.keep:
        keys = mx.concat(
            [
                self.keys[..., : self.keep, :],
                self.keys[..., (self.offset - (self.max_size - self.keep)) : self.offset, :],
            ],
            axis=2,
        )
        values = mx.concat(
            [
                self.values[..., : self.keep, :],
                self.values[..., (self.offset - (self.max_size - self.keep)) : self.offset, :],
            ],
            axis=2,
        )
        return keys, values
    else:
        return (
            self.keys[..., (self.offset - self.max_size) : self.offset, :],
            self.values[..., (self.offset - self.max_size) : self.offset, :],
        )


def apply_cache_patch() -> None:
    """Apply patches to RotatingKVCache for proper offset tracking.

    This function patches the following methods:
    - update_and_fetch: Properly sets _idx after update
    - state property: Returns correct key/value ranges based on offset

    The patch is idempotent and will only be applied once.
    """
    global _PATCHED
    if _PATCHED:
        return

    # Store original setter
    _original_state_setter = KVCache.state.fset

    # Apply patches
    RotatingKVCache.update_and_fetch = _new_update_and_fetch
    RotatingKVCache.state = property(_new_state_getter, _original_state_setter)
    RotatingKVCache.trim = KVCache.trim
    RotatingKVCache.is_trimmable = KVCache.is_trimmable

    _PATCHED = True
