"""Instrumented production pipeline: logs tokens, cost (USD) and milliseconds for every step of every query.

Steps:  preprocess (normalise + guards) -> [embed_query -> cache_lookup] -> retrieve -> generate (LLM) -> [cache_store]
mode="offline": local steps are timed for real; LLM/embedding latency uses a *simulated* model (LAT below) and tokens are estimated.
mode="openai" : real API calls; tokens come from response.usage and latency is wall-clock.
"""
import re, time, zlib
from dataclasses import replace
from rag_pipeline import FIXED, retrieve, _offline_generate, out_of_scope, REFUSAL, SYSTEM_V2
from handlers import precheck, MESSAGES
from cache import hash_embed
from pricing import cost_usd

LAT = dict(ttft_ms=380.0, per_in_tok_ms=0.12, per_out_tok_ms=14.0, embed_ms=90.0)   # assumed gpt-4o-mini-class latency (offline simulation only)
EMBED_MODEL = "text-embedding-3-small"

def est_tokens(text): return int(round(len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text)) * 1.15)) if text else 0
def _jit(key): return 0.85 + 0.3 * (zlib.crc32(key.encode()) % 1000) / 1000.0

class Pipeline:
    def __init__(self, cfg=FIXED, cache=None, gen_model="gpt-4o-mini", mode="offline"):
        self.cfg, self.cache, self.model, self.mode = replace(cfg, mode=mode), cache, gen_model, mode
        self._client = None
    def _oai(self):
        if self._client is None:
            from openai import OpenAI; self._client = OpenAI()
        return self._client

    # ---- step helpers: each returns result + a step record
    def _embed(self, text):
        t0 = time.perf_counter()
        if self.mode == "openai":
            r = self._oai().embeddings.create(model=EMBED_MODEL, input=text); vec, tin = r.data[0].embedding, r.usage.prompt_tokens
            ms = (time.perf_counter() - t0) * 1000
        else:
            vec, tin = hash_embed(text), est_tokens(text)
            ms = (time.perf_counter() - t0) * 1000 + LAT["embed_ms"] * _jit(text)
        return vec, dict(step="embed_query", ms=ms, tokens_in=tin, tokens_out=0, cost=cost_usd(EMBED_MODEL, tin), model=EMBED_MODEL)

    def _generate(self, text, retrieved):
        ctx = "\n".join(f"[{c['id']}] {c['text']}" for c, _ in retrieved); user = f"Context:\n{ctx}\n\nQuestion: {text}"
        t_sys, t_ctx, t_q = est_tokens(SYSTEM_V2), est_tokens(ctx), est_tokens(text); t0 = time.perf_counter()
        if self.mode == "openai":
            r = self._oai().chat.completions.create(model=self.model, temperature=0, messages=[{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": user}])
            ans, tin, tout = r.choices[0].message.content.strip(), r.usage.prompt_tokens, r.usage.completion_tokens
            ms = (time.perf_counter() - t0) * 1000; ttft = decode = None
        else:
            ans = REFUSAL if (self.cfg.abstain_guard and out_of_scope(text)) else _offline_generate(text, retrieved, self.cfg)
            tin, tout = t_sys + t_ctx + t_q + 8, est_tokens(ans)           # +8 = chat-format overhead
            real = (time.perf_counter() - t0) * 1000; j = _jit(text)
            ttft = (LAT["ttft_ms"] + LAT["per_in_tok_ms"] * tin) * j; decode = LAT["per_out_tok_ms"] * tout * j; ms = real + ttft + decode
        return ans, dict(step="generate", ms=ms, tokens_in=tin, tokens_out=tout, cost=cost_usd(self.model, tin, tout), model=self.model,
                         tok_system=t_sys, tok_context=t_ctx, tok_question=t_q, ttft_ms=ttft, decode_ms=decode)

    # ---- main entry
    def run(self, query):
        steps = []; t0 = time.perf_counter()
        text, handler, msg, _ = precheck(query)
        steps.append(dict(step="preprocess", ms=(time.perf_counter() - t0) * 1000, tokens_in=0, tokens_out=0, cost=0.0))
        if handler: return self._result(msg, [], [], steps, handler=handler)
        vec = None
        if self.cache is not None:
            vec, rec = self._embed(text); steps.append(rec)
            t1 = time.perf_counter(); hit, sim = self.cache.lookup(vec)
            steps.append(dict(step="cache_lookup", ms=(time.perf_counter() - t1) * 1000, tokens_in=0, tokens_out=0, cost=0.0))
            if hit: return self._result(hit["answer"], hit["context"], hit["retrieved_ids"], steps, cache_hit=True, sim=sim, cached_from=hit["question"])
        t1 = time.perf_counter(); retrieved = retrieve(text, self.cfg)
        steps.append(dict(step="retrieve", ms=(time.perf_counter() - t1) * 1000, tokens_in=0, tokens_out=0, cost=0.0))
        ans, rec = self._generate(text, retrieved); steps.append(rec)
        ctx, ids = [c["text"] for c, _ in retrieved], [c["id"] for c, _ in retrieved]
        if self.cache is not None and ans != REFUSAL:
            t1 = time.perf_counter(); self.cache.put(vec, dict(question=text, answer=ans, context=ctx, retrieved_ids=ids))
            steps.append(dict(step="cache_store", ms=(time.perf_counter() - t1) * 1000, tokens_in=0, tokens_out=0, cost=0.0))
        return self._result(ans, ctx, ids, steps)

    @staticmethod
    def _result(ans, ctx, ids, steps, handler=None, cache_hit=False, sim=None, cached_from=None):
        return dict(answer=ans, context=ctx, retrieved_ids=ids, handler=handler, steps=steps, cache_hit=cache_hit, cache_sim=sim, cached_from=cached_from,
                    cost_usd=sum(s["cost"] for s in steps), latency_ms=sum(s["ms"] for s in steps))
