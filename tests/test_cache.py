"""Run: python -m pytest tests -q   (or python tests/test_cache.py)  - exercises RedisStore against a minimal fake Redis client."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cache import SemanticCache, RedisStore, InMemoryStore, hash_embed

class FakeRedis:                      # just enough of redis-py for RedisStore
    def __init__(s): s.h, s.l, s.n = {}, {}, 0
    def incr(s, k): s.n += 1; return s.n
    def hset(s, key, mapping): s.h[key] = dict(mapping)
    def expire(s, key, ttl): pass
    def rpush(s, k, v): s.l.setdefault(k, []).append(v)
    def lrange(s, k, a, b): return list(s.l.get(k, []))
    def hmget(s, key, fields): d = s.h.get(key); return [d[f] if d else None for f in fields]
    def llen(s, k): return len(s.l.get(k, []))

def _roundtrip(store):
    c = SemanticCache(store, 0.92); q = "How long is the free trial?"
    assert c.lookup(hash_embed(q))[0] is None
    c.put(hash_embed(q), {"question": q, "answer": "14-day", "context": [], "retrieved_ids": []})
    hit, sim = c.lookup(hash_embed("how long is the free trial"))
    assert hit and hit["answer"] == "14-day" and sim >= 0.92
    assert c.lookup(hash_embed("What is the API rate limit on Scale?"))[0] is None

def test_memory(): _roundtrip(InMemoryStore())
def test_redis_store_logic(): _roundtrip(RedisStore(client=FakeRedis()))
if __name__ == "__main__": test_memory(); test_redis_store_logic(); print("cache tests passed")
