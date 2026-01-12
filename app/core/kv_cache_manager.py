"""KVCache management for prompt cache reuse across requests.

This module provides a cache manager that stores and reuses KVCache instances
to eliminate redundant computation for requests with shared prompt prefixes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class CacheEntry:
    """Represents a cached KVCache with metadata.

    Attributes
    ----------
    cache : Any
        The KVCache object from mlx_lm.
    token_ids : list[int]
        Full token sequence (prompt + generated tokens).
    last_used : datetime
        Timestamp of last access for LRU eviction.
    entry_id : int
        Unique identifier for this cache entry.
    is_locked : bool
        Whether the entry is currently in use (prevents reuse during generation).
    """

    cache: Any
    token_ids: list[int]
    last_used: datetime = field(default_factory=datetime.now)
    entry_id: int = 0
    is_locked: bool = False


class KVCacheManager:
    """Manages a pool of KVCache instances for reuse across requests.

    This manager implements prefix matching to find the best cache to reuse,
    LRU eviction when the pool is full, and entry locking to prevent
    concurrent access during generation.

    Parameters
    ----------
    max_cache_count : int
        Maximum number of cached prompts to store.
    min_prefix_length : int
        Minimum prefix length required for cache reuse. Defaults to 10.
    min_match_ratio : float
        Minimum match ratio (prefix_len / new_tokens_len) required for cache reuse.
        Prevents reusing caches with very low match ratios. Defaults to 0.1 (10%).
    """

    def __init__(
        self,
        max_cache_count: int,
        min_prefix_length: int = 10,
        min_reuse_ratio: float = 0.25,
    ) -> None:
        self.max_cache_count = max_cache_count
        self.min_prefix_length = min_prefix_length
        self.min_reuse_ratio = min_reuse_ratio
        self.entries: dict[int, CacheEntry] = {}
        self._next_entry_id = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

        logger.debug(
            f"KVCacheManager initialized: max_cache_count={max_cache_count}, "
            f"min_prefix_length={min_prefix_length}, min_reuse_ratio={min_reuse_ratio}"
        )

    async def find_best_match(
        self, token_ids: list[int]
    ) -> tuple[Any | None, int, int | None, int | None]:
        """Find cache with longest matching prefix (non-locked entries only).

        The cache is reusable if it shares a common prefix with the new token
        sequence. If the cached sequence is longer than the matching prefix,
        it will be trimmed before use.

        This implements the llama.cpp cache matching algorithm:
        1. Calculate reuse_ratio (f_keep) = prefix_len / len(cached_tokens)
        2. Calculate match_ratio (sim) = prefix_len / len(new_tokens)
        3. Filter by min_reuse_ratio (default 0.25)
        4. Select if both metrics are strictly better than current best

        Parameters
        ----------
        token_ids : list[int]
            Token sequence to match against cached sequences.

        Returns
        -------
        tuple[Any | None, int, int | None, int | None]
            A tuple of (cache, prefix_length, entry_id, cached_tokens_len).
            Returns (None, 0, None, None) if no suitable match found.
        """
        async with self._lock:
            best_entry: CacheEntry | None = None
            best_prefix_len = 0
            best_entry_id: int | None = None
            best_reuse_ratio = 0.0
            best_match_ratio = 0.0

            for entry in self.entries.values():
                # Skip locked entries (currently generating)
                if entry.is_locked:
                    continue

                cached_tokens = entry.token_ids

                # Log for debugging
                logger.info(
                    f"Comparing: cached={len(cached_tokens)} tokens, "
                    f"new={len(token_ids)} tokens, entry_id={entry.entry_id}"
                )

                # Find longest common prefix
                prefix_len = self._compute_prefix_length(cached_tokens, token_ids)

                # Calculate metrics
                # f_keep_cur: Reuse ratio of cached prompt
                reuse_ratio = prefix_len / len(cached_tokens) if len(cached_tokens) > 0 else 0.0
                # sim_cur: Similarity with new request
                match_ratio = prefix_len / len(token_ids) if len(token_ids) > 0 else 0.0

                if prefix_len > 0:
                    logger.info(
                        f"  -> Match: {prefix_len} tokens, "
                        f"reuse_ratio={reuse_ratio:.1%}, match_ratio={match_ratio:.1%}"
                    )

                # Threshold check (llama.cpp: f_keep_cur < 0.25)
                # We also keep min_prefix_length
                if prefix_len < self.min_prefix_length:
                    continue

                if reuse_ratio < self.min_reuse_ratio:
                    logger.info(
                        f"  -> Rejected: reuse_ratio {reuse_ratio:.1%} < "
                        f"min_reuse_ratio {self.min_reuse_ratio:.1%}"
                    )
                    continue

                # Find cache with better similarity (aligned with server-context.cpp slot selection)
                # Priority 1: Longest Common Prefix (Maximize tokens saved)
                # Priority 2: Higher Reuse Ratio (Tie-breaker: prefer cleaner cache)

                is_better = False
                if best_entry is None or prefix_len > best_prefix_len:
                    is_better = True
                elif prefix_len == best_prefix_len:
                    if reuse_ratio > best_reuse_ratio:
                        is_better = True

                if is_better:
                    best_entry = entry
                    best_prefix_len = prefix_len
                    best_entry_id = entry.entry_id
                    best_reuse_ratio = reuse_ratio
                    best_match_ratio = match_ratio
                    logger.info(
                        f"  -> New best match found: entry_id={entry.entry_id} (LCP={prefix_len})"
                    )

            if best_entry is not None:
                best_entry.last_used = datetime.now()
                best_entry.is_locked = True  # Lock during generation
                self._hits += 1
                logger.debug(
                    f"Cache hit: entry_id={best_entry_id}, "
                    f"prefix_len={best_prefix_len}/{len(token_ids)}"
                )
                cached_tokens_len = len(best_entry.token_ids)
                return best_entry.cache, best_prefix_len, best_entry_id, cached_tokens_len

            self._misses += 1
            logger.debug("Cache miss: no matching prefix found")
            return None, 0, None, None

    @staticmethod
    def _compute_prefix_length(seq1: list[int], seq2: list[int]) -> int:
        """Compute common prefix length between two token sequences.

        Parameters
        ----------
        seq1 : list[int]
            First token sequence.
        seq2 : list[int]
            Second token sequence.

        Returns
        -------
        int
            Length of the common prefix.
        """
        prefix_len = 0
        for t1, t2 in zip(seq1, seq2, strict=False):
            if t1 != t2:
                break
            prefix_len += 1
        return prefix_len

    async def save_cache(
        self,
        cache: Any,
        token_ids: list[int],
        entry_id: int | None = None,
    ) -> None:
        """Save cache to pool, evicting LRU if pool is full.

        Parameters
        ----------
        cache : Any
            KVCache object to save.
        token_ids : list[int]
            Complete token sequence (prompt + generated tokens).
        entry_id : int | None
            If provided, update this specific entry (reuse scenario).
            If None, create a new entry.
        """
        async with self._lock:
            logger.info(
                f"save_cache called: entry_id={entry_id}, tokens={len(token_ids)}, "
                f"current_entries={len(self.entries)}, max={self.max_cache_count}"
            )

            # Update existing entry if provided
            if entry_id is not None and entry_id in self.entries:
                entry = self.entries[entry_id]
                entry.cache = cache
                entry.token_ids = token_ids.copy()
                entry.last_used = datetime.now()
                entry.is_locked = False  # Unlock after save
                logger.info(f"Cache updated: entry_id={entry_id}, tokens={len(token_ids)}")
                return

            # Add new entry if pool not full
            if len(self.entries) < self.max_cache_count:
                new_entry_id = self._next_entry_id
                self._next_entry_id += 1
                self.entries[new_entry_id] = CacheEntry(
                    cache=cache,
                    token_ids=token_ids.copy(),
                    last_used=datetime.now(),
                    entry_id=new_entry_id,
                    is_locked=False,
                )
                logger.info(f"Cache created: entry_id={new_entry_id}, tokens={len(token_ids)}")
            else:
                # Evict LRU unlocked entry
                unlocked_entries = [e for e in self.entries.values() if not e.is_locked]
                if unlocked_entries:
                    lru_entry = min(unlocked_entries, key=lambda e: e.last_used)
                    lru_entry.cache = cache
                    lru_entry.token_ids = token_ids.copy()
                    lru_entry.last_used = datetime.now()
                    lru_entry.is_locked = False
                    logger.debug(
                        f"Cache evicted and replaced: entry_id={lru_entry.entry_id}, "
                        f"tokens={len(token_ids)}"
                    )
                else:
                    logger.warning(
                        "Cannot save cache: all entries are locked "
                        f"(max_cache_count={self.max_cache_count})"
                    )

    async def unlock_entry(self, entry_id: int | None) -> None:
        """Unlock an entry (called when generation fails or is cancelled).

        Parameters
        ----------
        entry_id : int | None
            The entry ID to unlock. If None, does nothing.
        """
        if entry_id is None:
            return

        async with self._lock:
            if entry_id in self.entries:
                self.entries[entry_id].is_locked = False
                logger.debug(f"Cache entry unlocked: entry_id={entry_id}")

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self.entries.clear()
            self._hits = 0
            self._misses = 0
            logger.info("KVCacheManager cleared")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns
        -------
        dict[str, Any]
            Dictionary containing cache statistics.
        """
        async with self._lock:
            total = self._hits + self._misses
            return {
                "total_entries": len(self.entries),
                "max_capacity": self.max_cache_count,
                "locked_entries": sum(1 for e in self.entries.values() if e.is_locked),
                "available_entries": sum(1 for e in self.entries.values() if not e.is_locked),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }
