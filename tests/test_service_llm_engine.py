import pytest
import mlx.core as mx
import threading
import time
from app.core.service_llm_engine import ServiceLLMEngine, GenerationContext


class MockDetokenizer:
    """Mock detokenizer that mimics mlx_lm StreamingDetokenizer interface."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the detokenizer state."""
        self.text = ""
        self.tokens = []
        self.offset = 0

    def add_token(self, token):
        """Add a token and update the text."""
        self.tokens.append(token)
        self.text += chr(token) if 32 <= token < 127 else "?"

    def finalize(self):
        """Finalize the detokenization (no-op for mock)."""
        # No-op for mock implementation

    @property
    def last_segment(self):
        """Return the last segment of readable text since last access."""
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self._detokenizer = MockDetokenizer()

    @property
    def detokenizer(self):
        # Return a fresh detokenizer instance (like the real TokenizerWrapper does)
        return MockDetokenizer()

    def encode(self, text, **kwargs):
        # Mock encoding: each char is a token id
        return [ord(c) for c in text]

    def decode(self, token_ids):
        # Mock decoding
        return "".join([chr(t) for t in token_ids])

class MockModel:
    def __init__(self):
        self.call_count = 0
        self.last_input_shape = None
    
    def __call__(self, input_ids, cache=None):
        self.call_count += 1
        self.last_input_shape = input_ids.shape
        # Return random logits: [Batch, Len, Vocab]
        # Vocab size = 100
        batch_size, seq_len = input_ids.shape
        return mx.random.normal((batch_size, seq_len, 100))

@pytest.fixture
def engine():
    model = MockModel()
    tokenizer = MockTokenizer()
    return ServiceLLMEngine(model, tokenizer)

def test_engine_initialization(engine):
    assert engine is not None
    assert isinstance(engine.model, MockModel)
    assert isinstance(engine.tokenizer, MockTokenizer)

def test_chunked_prefill_logic(engine):
    # Test that large prompts are chunked
    # Prompt length = 1000
    # Batch size in engine is hardcoded to 512
    prompt = mx.zeros((1, 1000), dtype=mx.int32)
    cache = [None] # Mock cache
    
    # We mock mx.eval to do nothing but track calls? 
    # Hard to mock mx.eval as it's a function in mlx.core.
    # But we can check model call count.
    
    # 1000 tokens. 
    # Chunk 1: 0-512 (512 tokens) -> model called
    # Chunk 2: 512-999 (487 tokens) -> prefill loop finishes?
    # Wait, the loop in ServiceLLMEngine is:
    # for i in range(0, N - 1, batch_size):
    #    chunk = prompt_tokens[:, i : min(i + batch_size, N - 1)]
    #    model(chunk)
    # 
    # N=1000. range(0, 999, 512) -> i=0, i=512.
    # i=0: chunk=0:512 (512 tokens). model called.
    # i=512: chunk=512:999 (487 tokens). model called.
    # Then final token (index 999) is passed to decode loop.
    
    # So model should be called 2 times in prefill + 1 time in decode (if max_tokens=1)
    
    list(engine.generate_stream(prompt, cache, max_tokens=1))
    
    # 2 prefill calls + 1 decode call = 3 calls
    assert engine.model.call_count >= 3

def test_cancellation(engine):
    prompt = mx.zeros((1, 10), dtype=mx.int32)
    cache = [None]
    context = GenerationContext()
    
    # Start generation in a separate thread and cancel it
    def run_gen():
        list(engine.generate_stream(prompt, cache, context=context, max_tokens=100))
        
    t = threading.Thread(target=run_gen)
    t.start()
    
    # Cancel immediately
    context.cancel()
    t.join()
    
    # Should not have generated 100 tokens (plus prefill)
    # 10 prompt tokens -> N=10. loop range(0, 9, 512) -> i=0 (chunk 0:9). 1 call.
    # Decode loop 100 calls.
    # Total calls if not cancelled: 1 + 100 = 101.
    
    # If cancelled immediately, it might be much less.
    assert engine.model.call_count < 50 

def test_stop_sequence(engine):
    # Mock sampler to return specific tokens to form a stop sequence
    # Stop sequence "STOP" -> ids [83, 84, 79, 80] (ASCII)
    
    # We need to mock the model or sampler to output these tokens.
    # Since ServiceLLMEngine calls sampler(logits), we can mock sampler.
    
    stop_tokens = [83, 84, 79, 80] # S, T, O, P
    
    class MockSampler:
        def __init__(self):
            self.count = 0
            
        def __call__(self, logits):
            # Return next token in sequence
            if self.count < len(stop_tokens):
                token = stop_tokens[self.count]
                self.count += 1
            else:
                token = 99 # eos
            return mx.array([token])
            
    sampler = MockSampler()
    prompt = mx.zeros((1, 5), dtype=mx.int32)
    cache = [None]
    
    # generate
    gen = engine.generate_stream(
        prompt, 
        cache, 
        max_tokens=10, 
        sampler=sampler, 
        stop_sequences=["STOP"]
    )
    
    output = "".join(list(gen))
    
    # Should return empty string because "STOP" matches and is consumed/removed?
    # Logic: 
    # S -> buffer "S", no match fully. yield?
    # StopSequenceHandler logic:
    # "S" -> in "STOP"? yes. buffer. return chunk="", stopped=False.
    # "ST" -> in "STOP"? yes. buffer.
    # ... "STOP" -> matches. return chunk="", stopped=True.
    
    # Wait, the naive implementation in ServiceLLMEngine:
    # self.current_text += new_text
    # if stop_seq in self.current_text:
    #    stop_index = ...
    #    final_chunk = self.current_text[:stop_index]
    #    return final_chunk, True
    
    # So "STOP" is found. stop_index starts at 0 (if "STOP" is at start of buffer).
    # final_chunk = ""[:0] -> "".
    # Returns "".
    
    assert output == ""
    
