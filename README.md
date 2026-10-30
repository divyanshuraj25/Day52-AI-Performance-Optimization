# Day 52 - Performance Optimisation (PayEasy Support Assistant)

| File | Purpose |
|---|---|
| `pipeline_perf.py` | Instrumented pipeline: tokens, USD cost and ms for every step of every query |
| `pricing.py` | Per-model token prices (verified Oct 2026; re-check before relying on them) |
| `cache.py` | Semantic cache: `RedisStore` (redis-py) / `InMemoryStore`, cosine threshold 0.92 |
| `run_perf.py` | Profile 20 queries -> projections -> cache -> retrieval threshold -> Day 50 eval gate -> `REPORT_DAY52.md` |
| `tests/test_cache.py` | Cache logic tests (RedisStore tested against a fake Redis client) |
| `app.py` | FastAPI `/ask` using the optimised pipeline (returns cache_hit, cost_usd, latency_ms) |
| `handlers.py`, `normalize.py`, `rag_pipeline.py`, ... | Day 50/51 code (guards, normalisation, RAG) |

```bash
python run_perf.py                       # offline (simulated LLM latency, estimated tokens)
python tests/test_cache.py
# real numbers + real Redis:
pip install -r requirements.txt
docker run -p 6379:6379 redis:7
export OPENAI_API_KEY=sk-...  REDIS_URL=redis://localhost:6379/0     # Windows: set ...
python run_perf.py --mode openai
uvicorn app:app --reload
```
