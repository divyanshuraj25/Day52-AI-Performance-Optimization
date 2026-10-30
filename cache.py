"""Semantic response cache.  embed(query) -> compare with cached query embeddings -> return cached answer if cosine >= 0.92.

Stores: RedisStore (production, redis-py) and InMemoryStore (tests / no Redis available).
Both expose add(vec, payload) and scan() -> iterable of (vec, payload).
For > ~10k cached entries replace the linear scan with a RediSearch HNSW vector index (same interface).
"""
import json, math, re, time, zlib
from array import array

THRESHOLD = 0.92

def hash_embed(text, dim=512):
    """Deterministic offline embedding (signed feature hashing of words, bigrams, char-trigrams). Real mode uses text-embedding-3-small."""
    t = text.lower(); words = re.findall(r"[a-z0-9$%.]+", t)
    feats = [(w, 1.0) for w in words] + [(a + "_" + b, 1.0) for a, b in zip(words, words[1:])] + [(t[i:i + 3], 0.5) for i in range(len(t) - 2)]
    v = [0.0] * dim
    for f, w in feats:
        h = zlib.crc32(f.encode()); v[h % dim] += w if (h >> 16) & 1 else -w
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) or 1.0; nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

class InMemoryStore:
    def __init__(self): self.items = []
    def add(self, vec, payload): self.items.append((vec, payload))
    def scan(self): return list(self.items)
    def __len__(self): return len(self.items)

class RedisStore:
    """Entries are Redis hashes  semcache:<n>  {emb: float32 bytes, payload: json}  with a TTL; ids tracked in list semcache:ids."""
    def __init__(self, client=None, url="redis://localhost:6379/0", ttl_seconds=24 * 3600, prefix="semcache"):
        if client is None:
            import redis; client = redis.Redis.from_url(url)
        self.r, self.ttl, self.p = client, ttl_seconds, prefix
    def add(self, vec, payload):
        key = f"{self.p}:{self.r.incr(self.p + ':seq')}"
        self.r.hset(key, mapping={"emb": array("f", vec).tobytes(), "payload": json.dumps(payload)})
        self.r.expire(key, self.ttl); self.r.rpush(self.p + ":ids", key)
    def scan(self):
        for key in self.r.lrange(self.p + ":ids", 0, -1):
            emb, payload = self.r.hmget(key, ["emb", "payload"])
            if emb is None: continue                      # expired by TTL
            a = array("f"); a.frombytes(emb); yield list(a), json.loads(payload)
    def __len__(self): return self.r.llen(self.p + ":ids")

class SemanticCache:
    def __init__(self, store=None, threshold=THRESHOLD):
        self.store, self.threshold = store or InMemoryStore(), threshold
        self.hits = self.misses = 0
    def lookup(self, vec):
        """-> (payload | None, best_similarity)"""
        best, best_sim = None, -1.0
        for v, payload in self.store.scan():
            s = cosine(vec, v)
            if s > best_sim: best, best_sim = payload, s
        if best is not None and best_sim >= self.threshold:
            self.hits += 1; return best, best_sim
        self.misses += 1; return None, best_sim
    def put(self, vec, payload): self.store.add(vec, payload)
