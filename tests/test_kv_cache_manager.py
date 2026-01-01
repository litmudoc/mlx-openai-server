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

@pytest.mark.asyncio
async def test_find_best_match_llama_cpp_logic():
    # Initialize manager with defaults (min_reuse_ratio=0.25)
    manager = KVCacheManager(max_cache_count=5, min_prefix_length=10, min_reuse_ratio=0.25)
    
    # Helper to create entry
    def create_entry(eid, tokens):
        return CacheEntry(
            cache=MagicMock(),
            token_ids=tokens,
            last_used=datetime.now(),
            entry_id=eid,
            is_locked=False
        )
    
    # Verify strict improvement logic.
    # Case: A (f=0.9, sim=0.5) vs B (f=0.8, sim=0.9).
    # A should be kept if seen first.
    
    manager.entries.clear()
    
    # Cache A: 100 tokens.
    # Cache B: 100 tokens.
    # New: 200 tokens.
    
    # A matches 90. reuse=0.9. sim=0.45 (req=200).
    # B matches 180 (Wait, B only has 100 tokens? No, let's say B has 200 tokens)
    
    # Cache A: [0..99]
    # Cache B: [0..89, 1000..1109] (Length 200).
    # New: [0..89, 1000..1109]
    
    # A matches 90. reuse=0.9. sim=90/200 = 0.45.
    # B matches 200. reuse=1.0. sim=1.0.
    
    # B dominates A.
    # If A first: Best=(0.9, 0.45).
    # Then B: (1.0 > 0.9) AND (1.0 > 0.45). Update -> B.
    # Result B.
    
    manager.entries[0] = create_entry(0, list(range(100))) # 0..99
    
    request_tokens = list(range(200))
    
    # Cache A: 0..89 matches. 90..99 mismatch.
    # reuse=90/100 = 0.9. sim=90/200=0.45.
    cache_A_tokens = list(range(90)) + [999]*10
    manager.entries[0] = create_entry(0, cache_A_tokens)
    
    # Cache B: 0..199 matches.
    # reuse=200/200 = 1.0. sim=200/200=1.0.
    cache_B_tokens = list(range(200))
    manager.entries[1] = create_entry(1, cache_B_tokens)
    
    # B strictly dominates A. Should pick B regardless of order.
    cache, prefix, eid = await manager.find_best_match(request_tokens)
    assert eid == 1
    assert prefix == 200

    # Now non-dominating case.
    # Cache X: 100 tokens. Match 90. reuse=0.9. sim=0.45 (req=200).
    # Cache Y: 200 tokens. Match 100. reuse=0.5. sim=0.5.
    
    # X: (0.9, 0.45)
    # Y: (0.5, 0.5)
    
    # X has better reuse. Y has better sim.
    # If X first: Y is NOT strictly better (0.5 < 0.9). Keep X.
    # If Y first: X is NOT strictly better (0.45 < 0.5). Keep Y.
    
    manager.entries.clear()
    
    # X
    cache_X_tokens = list(range(90)) + [888]*10 # 100 total
    manager.entries[10] = create_entry(10, cache_X_tokens)
    
    # Y
    cache_Y_tokens = list(range(100)) + [777]*100 # 200 total
    manager.entries[11] = create_entry(11, cache_Y_tokens)
    
    request_tokens = list(range(100)) + [999]*100 # 200 total
    
    # X matches 90. reuse=0.9. sim=0.45.
    # Y matches 100. reuse=0.5. sim=0.5.
    
    # Order X, then Y.
    cache, prefix, eid = await manager.find_best_match(request_tokens)
    assert eid == 10 # Picks X because Y is not strictly better
    
    # Order Y, then X.
    manager.entries.clear()
    manager.entries[11] = create_entry(11, cache_Y_tokens)
    manager.entries[10] = create_entry(10, cache_X_tokens)
    
    cache, prefix, eid = await manager.find_best_match(request_tokens)
    assert eid == 11 # Picks Y because X is not strictly better
    
    # Verify reuse ratio threshold
    manager.entries.clear()
    manager.entries[2] = create_entry(2, list(range(100)))
    
    new_tokens_short = list(range(20)) + [999]*80
    # reuse = 20/100 = 0.2 < 0.25
    cache, prefix, eid = await manager.find_best_match(new_tokens_short)
    assert cache is None
