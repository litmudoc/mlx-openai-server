import pytest
from app.core.kv_cache_manager import KVCacheManager, CacheEntry
from datetime import datetime
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_find_best_match_with_trim():
    # Initialize manager
    manager = KVCacheManager(max_cache_count=5, min_prefix_length=10)
    
    # Create a mock cache entry with 100 tokens
    cached_tokens = list(range(100))
    mock_cache = [MagicMock() for _ in range(2)]
    for m in mock_cache:
        m.offset = 100
        
    entry = CacheEntry(
        cache=mock_cache,
        token_ids=cached_tokens,
        last_used=datetime.now(),
        entry_id=0,
        is_locked=False
    )
    manager.entries[0] = entry
    
    # Case 1: New request is shorter but matches prefix
    new_tokens = list(range(50))
    cache, prefix_len, entry_id = await manager.find_best_match(new_tokens)
    
    assert cache == mock_cache
    assert prefix_len == 50
    assert entry_id == 0

@pytest.mark.asyncio
async def test_find_best_match_longer_request():
    # Initialize manager
    manager = KVCacheManager(max_cache_count=5, min_prefix_length=10)
    
    # Create a mock cache entry with 50 tokens
    cached_tokens = list(range(50))
    mock_cache = [MagicMock() for _ in range(2)]
    for m in mock_cache:
        m.offset = 50
        
    entry = CacheEntry(
        cache=mock_cache,
        token_ids=cached_tokens,
        last_used=datetime.now(),
        entry_id=0,
        is_locked=False
    )
    manager.entries[0] = entry
    
    # Case 2: New request is longer and matches prefix
    new_tokens = list(range(100))
    cache, prefix_len, entry_id = await manager.find_best_match(new_tokens)
    
    assert cache == mock_cache
    assert prefix_len == 50
    assert entry_id == 0

@pytest.mark.asyncio
async def test_find_best_match_mismatch():
    # Initialize manager
    manager = KVCacheManager(max_cache_count=5, min_prefix_length=10)
    
    # Create a mock cache entry
    cached_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mock_cache = [MagicMock()]
    
    entry = CacheEntry(
        cache=mock_cache,
        token_ids=cached_tokens,
        last_used=datetime.now(),
        entry_id=0,
        is_locked=False
    )
    manager.entries[0] = entry
    
    # Case 3: Mismatch at position 5
    new_tokens = [1, 2, 3, 4, 5, 99, 7, 8, 9, 10, 11, 12]
    cache, prefix_len, entry_id = await manager.find_best_match(new_tokens)
    
    # Prefix length is 5, which is less than min_prefix_length (10)
    assert cache is None
    assert prefix_len == 0
    
    # Case 4: Mismatch at position 11
    new_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99]
    cache, prefix_len, entry_id = await manager.find_best_match(new_tokens)
    
    assert cache == mock_cache
    assert prefix_len == 11
    assert entry_id == 0
