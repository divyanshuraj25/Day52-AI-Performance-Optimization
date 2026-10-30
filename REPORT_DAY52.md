# Day 52 - Performance Optimisation: PayEasy Support Assistant

Run mode: **offline** | 20261029_111638

> **Offline mode:** local step timings (normalise, guards, BM25, cache lookup) are real measurements. LLM and embedding **latency is simulated** (gpt-4o-mini-class: 380 ms TTFT + 0.12 ms/input token + 14 ms/output token; embeddings 90 ms) and **token counts are estimated** (~1.15 x regex tokens). Prices are real. Re-run with `--mode openai` for measured usage/latency. Cache hit-rate comes from a simulated 300-request Zipf traffic mix, so it depends on assumptions about real traffic.

## 1. Pipeline profile (20 sample queries, baseline)

Generation model: `gpt-4o-mini`. Avg tokens per call: 249 in / 46 out.

| Step | Avg ms | Avg tokens in | Avg tokens out | Avg cost (USD) |
|---|---|---|---|---|
| preprocess | 0.14 | 0 | 0 | 0.00000000 |
| retrieve | 0.16 | 0 | 0 | 0.00000000 |
| generate | 1089.03 | 249.2 | 45.5 | 0.00006472 |

**Average cost per query: $0.0000647 | Average latency: 1089 ms**

### Monthly cost projection (baseline, 30 days)

| Queries/day | Monthly cost |
|---|---|
| 1000 | $1.94 |
| 10000 | $19.42 |
| 100000 | $194.15 |

## 2. Top 3 cost drivers

1. completion (output): 42% of cost - output tokens cost 4x input tokens ($0.6/1M vs $0.15/1M) and the bot writes up to 3 sentences per answer.
2. retrieved context (input): 28% of cost - top_k=3 chunks (~122 tokens) are re-sent on every call even when one chunk holds the answer.
3. system prompt (input): 24% of cost - the 103-token v2 safety/format prompt is identical on every call but is billed every time.

## 3. Top 2 latency bottlenecks

1. LLM decode (output tokens): 662 ms (61% of latency) - decoding is sequential at ~14ms per output token, so every extra sentence adds latency.
2. LLM time-to-first-token (prefill + network): 426 ms (39% of latency) - each call pays a fixed network + queueing + prefill cost before the first token, growing with the prompt length.

## 4. Optimisation 1 - semantic response cache (Redis)

`cache.py`: embed query -> cosine against cached query embeddings -> return cached answer if similarity >= **0.92**. `RedisStore` keeps each entry as a Redis hash (float32 embedding + JSON payload) with a 24h TTL; `InMemoryStore` is used when `REDIS_URL` is not set. Only real answers are cached (refusals and handler replies are never cached). Cost of the cache: one embedding call per request (~$0.0000002).

- Traffic hit rate: **82%** | false hits (wrong question served): **0**
- Highest similarity between two *different* eval questions: [0.607, 'E04', 'H01'] (must stay < 0.92 to avoid false hits)

## 5. Optimisation 2 - stricter retrieval threshold (prompt-size reduction)

Keep only retrieved chunks scoring >= X x best score. Sweep on the Day 50 suite:

| min_rel_score | Eval overall | Avg context tokens | Avg cost/query |
|---|---|---|---|
| 0.0 | 0.84 | 120.8 | 0.0000628 |
| 0.4 | 0.83 | 73.9 | 0.0000543 |
| 0.5 | 0.85 | 69 | 0.0000534 |
| 0.6 | 0.85 | 66.4 | 0.0000530 |
| 0.7 | 0.85 | 58.3 | 0.0000517 |
| 0.8 | 0.84 | 51 | 0.0000502 |

Chosen: **0.8** (most aggressive value with no quality loss on any dimension).

## 6. Day 50 eval after each optimisation (stop rule: any dimension drops > 0.3)

| Stage | correctness | groundedness | completeness | refusal_appropriateness | overall | Worst drop vs baseline | Stop? |
|---|---|---|---|---|---|---|---|
| baseline | 0.733 | 1.0 | 0.733 | 1.0 | 0.84 | - | - |
| opt1 semantic cache (cold pass) | 0.733 | 1.0 | 0.733 | 1.0 | 0.84 | 0.0 (correctness) | ok |
| opt1 semantic cache (warm reworded replay) | 0.733 | 1.0 | 0.733 | 1.0 | 0.84 | 0.0 (correctness) | ok |
| opt1+opt2 (cache + retrieval threshold 0.8) | 0.733 | 1.0 | 0.733 | 1.0 | 0.84 | 0.0 (correctness) | ok |

## 7. Cost - latency - quality comparison (300-request simulated traffic)

| Version | Avg cost/query | Avg latency (ms) | p95 latency (ms) | Cache hit rate | Day 50 eval (overall) |
|---|---|---|---|---|---|
| Baseline | $0.0000636 | 1056 | 1208 | 0% | 0.84 |
| + Semantic cache | $0.0000112 | 270 | 1232 | 82% | 0.84 |
| + Cache + retrieval threshold (all) | $0.0000089 | 259 | 1142 | 82% | 0.84 |

### Sensitivity: the 82% hit rate comes from a simulated 30-question FAQ mix and is optimistic

| Cache hit rate | Avg cost/query | Avg latency (ms) | Monthly cost @100k/day |
|---|---|---|---|
| 0% | $0.0000468 | 978 | $140.43 |
| 20% | $0.0000375 | 802 | $112.59 |
| 40% | $0.0000282 | 626 | $84.75 |
| 60% | $0.0000190 | 450 | $56.91 |
| 80% | $0.0000097 | 274 | $29.07 |

**Net effect:** cost 86% lower, average latency 75% lower, Day 50 eval 0.84 -> 0.84.

### Monthly projection after optimisation

| Queries/day | Baseline | + Cache | + Cache + threshold |
|---|---|---|---|
| 1,000 | $1.91 | $0.34 | $0.27 |
| 10,000 | $19.08 | $3.36 | $2.68 |
| 100,000 | $190.79 | $33.55 | $26.75 |

### Not implemented (next candidates)

- Shorten the system prompt (~24% of cost, billed identically on every call); verify with the eval suite in `--mode openai` because prompt wording affects real LLM behaviour.
- Parallelise independent steps (embedding + BM25) with `asyncio.gather` once a dense retriever is added.
- Stream the answer to cut perceived latency.