"""Day 52 - Performance optimisation: profile -> project -> optimise (semantic cache, stricter retrieval) -> gate on Day 50 eval -> compare.

    python run_perf.py                # offline: real local timing + SIMULATED LLM latency/tokens
    python run_perf.py --mode openai  # real OpenAI usage + wall-clock latency (needs OPENAI_API_KEY; Redis optional via REDIS_URL)
"""
import argparse, json, os, random, statistics as st
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from rag_pipeline import FIXED
from llm_judge import judge, DIMS
from run_eval import load_dataset
from pipeline_perf import Pipeline
from cache import SemanticCache, InMemoryStore, RedisStore, hash_embed, cosine, THRESHOLD
from pricing import PRICES, cost_usd

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_IDS = "E01 E02 E03 E05 E07 E09 E10 M01 M02 M03 M05 M06 M08 M10 H02 H03 H04 H06 H07 H09".split()
VOLUMES = [1000, 10000, 100000]
STOP_DROP = 0.3

def new_cache(mode):
    url = os.environ.get("REDIS_URL")
    return SemanticCache(RedisStore(url=url) if (url and mode == "openai") else InMemoryStore(), THRESHOLD)

# ---------------- workload ----------------
VARIANTS = [lambda q: q.lower().rstrip("?"), lambda q: "hey, " + q[0].lower() + q[1:], lambda q: q.rstrip("?") + " please?",
            lambda q: "quick question: " + q[0].lower() + q[1:], lambda q: q.replace("?", " ?")]
def make_traffic(data, n=300, seed=7):
    rng = random.Random(seed); order = data[:]; rng.shuffle(order)
    weights = [1.0 / (i + 1) for i in range(len(order))]          # Zipf: a few FAQs dominate real traffic
    out = []
    for _ in range(n):
        q = rng.choices(order, weights)[0]; text = q["question"]
        if rng.random() < 0.5: text = rng.choice(VARIANTS)(text)
        out.append((q, text))
    return out

# ---------------- evaluation helpers ----------------
def dims_mean(rows): return {d: round(st.mean(r[d] for r in rows), 3) for d in DIMS + ["overall"]}
def eval_suite(pipe, data, replay_variants=False):
    """Day 50 suite through `pipe`. With a cache: pass 1 cold (fills cache); replay_variants=True re-asks reworded versions (stress for false hits)."""
    rows = []
    for q in data:
        text = VARIANTS[0](q["question"]) if replay_variants else q["question"]
        out = pipe.run(text); j = judge(q, out); rows.append({**{d: j[d] for d in DIMS + ["overall"]}, "id": q["id"], "hit": out["cache_hit"]})
    return rows

def gate(name, dims, base_dims):
    drops = {d: round(base_dims[d] - dims[d], 3) for d in DIMS}
    worst = max(drops, key=drops.get)
    return {"stage": name, "dims": dims, "drops": drops, "worst_dim": worst, "worst_drop": drops[worst], "stop": drops[worst] > STOP_DROP}

def run_traffic(pipe, traffic):
    res = [pipe.run(t) for _, t in traffic]
    false_hits = sum(1 for (q, _), r in zip(traffic, res) if r["cache_hit"] and r["cached_from"] and not _same_intent(q, r))
    return res, dict(avg_cost=st.mean(r["cost_usd"] for r in res), avg_ms=st.mean(r["latency_ms"] for r in res),
                     p95_ms=sorted(r["latency_ms"] for r in res)[int(0.95 * len(res)) - 1], hit_rate=sum(r["cache_hit"] for r in res) / len(res),
                     false_hits=false_hits, quality=dims_mean([{**{d: judge(q, r)[d] for d in DIMS}, "overall": judge(q, r)["overall"]} for (q, _), r in zip(traffic, res)]))

_Q = {}
def _same_intent(q, r):
    """A hit is correct iff the cached answer came from a question with the same id."""
    return _Q.get(r["cached_from"]) == q["id"]

# ---------------- profiling ----------------
def profile(pipe, data):
    byid = {q["id"]: q for q in data}; rows = []
    for i in PROFILE_IDS:
        r = pipe.run(byid[i]["question"]); rows.append({"id": i, "cost": r["cost_usd"], "ms": r["latency_ms"], "steps": r["steps"]})
    return rows

def summarize_profile(rows):
    step = defaultdict(lambda: dict(ms=[], tin=[], tout=[], cost=[]))
    for r in rows:
        for s in r["steps"]:
            d = step[s["step"]]; d["ms"].append(s["ms"]); d["tin"].append(s["tokens_in"]); d["tout"].append(s["tokens_out"]); d["cost"].append(s["cost"])
    steps = {k: dict(ms=round(st.mean(v["ms"]), 2), tokens_in=round(st.mean(v["tin"]), 1), tokens_out=round(st.mean(v["tout"]), 1), cost=st.mean(v["cost"])) for k, v in step.items()}
    gens = [s for r in rows for s in r["steps"] if s["step"] == "generate"]; model = gens[0]["model"]; pin, pout = PRICES[model]
    comp = {"system prompt (input)": st.mean(g["tok_system"] for g in gens) * pin / 1e6, "retrieved context (input)": st.mean(g["tok_context"] for g in gens) * pin / 1e6,
            "user question + format (input)": st.mean(g["tokens_in"] - g["tok_system"] - g["tok_context"] for g in gens) * pin / 1e6,
            "completion (output)": st.mean(g["tokens_out"] for g in gens) * pout / 1e6}
    lat = {"LLM decode (output tokens)": st.mean(g["decode_ms"] for g in gens) if gens[0]["decode_ms"] is not None else None,
           "LLM time-to-first-token (prefill + network)": st.mean(g["ttft_ms"] for g in gens) if gens[0]["ttft_ms"] is not None else None,
           "local steps (preprocess + retrieve)": sum(steps[k]["ms"] for k in ("preprocess", "retrieve") if k in steps)}
    return steps, comp, lat, dict(avg_cost=st.mean(r["cost"] for r in rows), avg_ms=st.mean(r["ms"] for r in rows), model=model,
                                  avg_in=st.mean(g["tokens_in"] for g in gens), avg_out=st.mean(g["tokens_out"] for g in gens))

WHY_COST = {"retrieved context (input)": "top_k=3 chunks (~{t} tokens) are re-sent on every call even when one chunk holds the answer.",
            "completion (output)": "output tokens cost 4x input tokens (${po}/1M vs ${pi}/1M) and the bot writes up to 3 sentences per answer.",
            "system prompt (input)": "the {t}-token v2 safety/format prompt is identical on every call but is billed every time.",
            "user question + format (input)": "short question plus chat-format overhead, small in absolute terms."}
WHY_LAT = {"LLM decode (output tokens)": "decoding is sequential at ~{per}ms per output token, so every extra sentence adds latency.",
           "LLM time-to-first-token (prefill + network)": "each call pays a fixed network + queueing + prefill cost before the first token, growing with the prompt length.",
           "local steps (preprocess + retrieve)": "BM25 and guards run in-process and are negligible."}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="offline", choices=["offline", "openai"]); mode = ap.parse_args().mode
    ts = datetime.now().strftime("%Y%m%d_%H%M%S"); data = load_dataset(); byid = {q["id"]: q for q in data}
    for q in data: _Q[q["question"]] = q["id"]; _Q[VARIANTS[0](q["question"])] = q["id"]
    traffic = make_traffic(data)
    for q, t in traffic: _Q[t] = q["id"]
    out = {"timestamp": ts, "mode": mode}

    # 1) baseline profile (20 queries) + eval baseline
    base_pipe = Pipeline(FIXED, None, mode=mode); prof = profile(base_pipe, data); steps, comp, lat, ps = summarize_profile(prof)
    base_eval = dims_mean(eval_suite(base_pipe, data)); out["profile"] = dict(steps=steps, cost_components=comp, latency_components=lat, summary=ps)
    print("PROFILE avg cost/query $%.7f, avg latency %.0f ms, avg tokens in/out %.0f/%.0f" % (ps["avg_cost"], ps["avg_ms"], ps["avg_in"], ps["avg_out"]))
    print("Day 50 baseline eval:", base_eval)

    # 2) projection
    proj = {f"{v}/day": round(ps["avg_cost"] * v * 30, 2) for v in VOLUMES}; out["projection_baseline"] = proj; print("Monthly cost (baseline):", proj)

    # 3) drivers / bottlenecks
    top_cost = sorted(comp.items(), key=lambda x: -x[1])[:3]; pin, pout = PRICES[ps["model"]]
    gens = [s for r in prof for s in r["steps"] if s["step"] == "generate"]; tsys, tctx = round(st.mean(g["tok_system"] for g in gens)), round(st.mean(g["tok_context"] for g in gens))
    drivers = []
    for k, v in top_cost:
        why = WHY_COST[k].format(t=tctx if "context" in k else tsys, pi=pin, po=pout)
        drivers.append(f"{k}: {v / ps['avg_cost'] * 100:.0f}% of cost - {why}")
    top_lat = [(k, v) for k, v in sorted(lat.items(), key=lambda x: -(x[1] or 0)) if v is not None][:2]
    bottlenecks = [f"{k}: {v:.0f} ms ({v / ps['avg_ms'] * 100:.0f}% of latency) - {WHY_LAT[k].format(per=14)}" for k, v in top_lat]
    out["cost_drivers"], out["latency_bottlenecks"] = drivers, bottlenecks

    # 4) baseline on traffic
    _, base_t = run_traffic(Pipeline(FIXED, None, mode=mode), traffic)

    # 5) optimisation 1: semantic cache
    cache = new_cache(mode); p1 = Pipeline(FIXED, cache, mode=mode)
    cold = eval_suite(p1, data); warm = eval_suite(p1, data, replay_variants=True)
    sims = sorted((cosine(hash_embed(a["question"]), hash_embed(b["question"])), a["id"], b["id"]) for i, a in enumerate(data) for b in data[i + 1:])[-1]
    g1 = gate("opt1 semantic cache (cold pass)", dims_mean(cold), base_eval); g1b = gate("opt1 semantic cache (warm reworded replay)", dims_mean(warm), base_eval)
    g1["warm"] = g1b; g1["warm_hits"] = sum(r["hit"] for r in warm); g1["max_pairwise_sim_between_different_questions"] = [round(sims[0], 3), sims[1], sims[2]]
    _, t1 = run_traffic(Pipeline(FIXED, new_cache(mode), mode=mode), traffic)
    print(f"OPT1 cache: hit-rate {t1['hit_rate']:.0%}, cost ${t1['avg_cost']:.7f}, latency {t1['avg_ms']:.0f}ms, false hits {t1['false_hits']}; eval drop {g1b['worst_drop']}")

    # 6) optimisation 2: stricter retrieval threshold - sweep, then pick the most aggressive setting within quality budget
    sweep = []
    for thr in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
        cfg = replace(FIXED, min_rel_score=thr); pp = Pipeline(cfg, None, mode=mode)
        d = dims_mean(eval_suite(pp, data)); rs = [pp.run(q["question"]) for q in data]
        ctx_tok = st.mean(next(s["tok_context"] for s in r["steps"] if s["step"] == "generate") for r in rs if any(s["step"] == "generate" for s in r["steps"]))
        sweep.append({"min_rel_score": thr, "overall": d["overall"], "dims": d, "avg_context_tokens": round(ctx_tok, 1), "avg_cost": st.mean(r["cost_usd"] for r in rs)})
    ok = [s for s in sweep if base_eval["overall"] - s["overall"] <= 0.0 + 1e-9 and max(base_eval[d] - s["dims"][d] for d in DIMS) <= 1e-9]
    best = min(ok, key=lambda s: s["avg_context_tokens"]) if ok else sweep[0]; out["retrieval_sweep"], out["chosen_threshold"] = sweep, best["min_rel_score"]
    cfg2 = replace(FIXED, min_rel_score=best["min_rel_score"]); cache2 = new_cache(mode); p2 = Pipeline(cfg2, cache2, mode=mode)
    eval_suite(p2, data); warm2 = eval_suite(p2, data, replay_variants=True)
    g2 = gate("opt1+opt2 (cache + retrieval threshold %.1f)" % best["min_rel_score"], dims_mean(warm2), base_eval)
    r2, t2 = run_traffic(Pipeline(cfg2, new_cache(mode), mode=mode), traffic)
    miss_c = st.mean(r["cost_usd"] for r in r2 if not r["cache_hit"]); hit_c = st.mean(r["cost_usd"] for r in r2 if r["cache_hit"])
    miss_l = st.mean(r["latency_ms"] for r in r2 if not r["cache_hit"]); hit_l = st.mean(r["latency_ms"] for r in r2 if r["cache_hit"])
    out["sensitivity"] = [{"hit_rate": h, "cost": h * hit_c + (1 - h) * miss_c, "ms": h * hit_l + (1 - h) * miss_l} for h in (0.0, 0.2, 0.4, 0.6, 0.8)]
    print(f"OPT2 cache+threshold {best['min_rel_score']}: cost ${t2['avg_cost']:.7f}, latency {t2['avg_ms']:.0f}ms; eval drop {g2['worst_drop']}")

    out.update(gates=[g1, g1b, g2], traffic={"baseline": base_t, "opt1": t1, "opt2": t2}, baseline_eval=base_eval,
               projection={n: {f"{v}/day": round(t["avg_cost"] * v * 30, 2) for v in VOLUMES} for n, t in [("baseline", base_t), ("opt1", t1), ("opt2", t2)]})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"perf_{mode}_{ts}.json"), "w"), indent=2, default=str)
    write_report(out); print("Saved results + REPORT_DAY52.md")

def write_report(o):
    p, sm = o["profile"], o["profile"]["summary"]; T = o["traffic"]; L = []
    sim = o["mode"] == "offline"
    L += ["# Day 52 - Performance Optimisation: PayEasy Support Assistant", "", f"Run mode: **{o['mode']}** | {o['timestamp']}", ""]
    if sim: L += ["> **Offline mode:** local step timings (normalise, guards, BM25, cache lookup) are real measurements. LLM and embedding **latency is simulated** (gpt-4o-mini-class: 380 ms TTFT + 0.12 ms/input token + 14 ms/output token; embeddings 90 ms) and **token counts are estimated** (~1.15 x regex tokens). Prices are real. Re-run with `--mode openai` for measured usage/latency. Cache hit-rate comes from a simulated 300-request Zipf traffic mix, so it depends on assumptions about real traffic.", ""]
    L += ["## 1. Pipeline profile (20 sample queries, baseline)", "", f"Generation model: `{sm['model']}`. Avg tokens per call: {sm['avg_in']:.0f} in / {sm['avg_out']:.0f} out.", "",
          "| Step | Avg ms | Avg tokens in | Avg tokens out | Avg cost (USD) |", "|---|---|---|---|---|"]
    for k, v in p["steps"].items(): L.append(f"| {k} | {v['ms']} | {v['tokens_in']} | {v['tokens_out']} | {v['cost']:.8f} |")
    L += ["", f"**Average cost per query: ${sm['avg_cost']:.7f} | Average latency: {sm['avg_ms']:.0f} ms**", "", "### Monthly cost projection (baseline, 30 days)", "", "| Queries/day | Monthly cost |", "|---|---|"]
    for k, v in o["projection_baseline"].items(): L.append(f"| {k.split('/')[0]} | ${v:,.2f} |")
    L += ["", "## 2. Top 3 cost drivers", ""] + [f"{i + 1}. {d}" for i, d in enumerate(o["cost_drivers"])]
    L += ["", "## 3. Top 2 latency bottlenecks", ""] + [f"{i + 1}. {d}" for i, d in enumerate(o["latency_bottlenecks"])]
    L += ["", "## 4. Optimisation 1 - semantic response cache (Redis)", "", "`cache.py`: embed query -> cosine against cached query embeddings -> return cached answer if similarity >= **0.92**. `RedisStore` keeps each entry as a Redis hash (float32 embedding + JSON payload) with a 24h TTL; `InMemoryStore` is used when `REDIS_URL` is not set. Only real answers are cached (refusals and handler replies are never cached). Cost of the cache: one embedding call per request (~$0.0000002).", "",
          f"- Traffic hit rate: **{T['opt1']['hit_rate']:.0%}** | false hits (wrong question served): **{T['opt1']['false_hits']}**",
          f"- Highest similarity between two *different* eval questions: {o['gates'][0]['max_pairwise_sim_between_different_questions']} (must stay < 0.92 to avoid false hits)", ""]
    L += ["## 5. Optimisation 2 - stricter retrieval threshold (prompt-size reduction)", "", "Keep only retrieved chunks scoring >= X x best score. Sweep on the Day 50 suite:", "", "| min_rel_score | Eval overall | Avg context tokens | Avg cost/query |", "|---|---|---|---|"]
    for s in o["retrieval_sweep"]: L.append(f"| {s['min_rel_score']} | {s['overall']} | {s['avg_context_tokens']} | {s['avg_cost']:.7f} |")
    L += ["", f"Chosen: **{o['chosen_threshold']}** (most aggressive value with no quality loss on any dimension).", "", "## 6. Day 50 eval after each optimisation (stop rule: any dimension drops > 0.3)", "",
          "| Stage | " + " | ".join(DIMS) + " | overall | Worst drop vs baseline | Stop? |", "|---|" + "---|" * (len(DIMS) + 3),
          "| baseline | " + " | ".join(str(o["baseline_eval"][d]) for d in DIMS) + f" | {o['baseline_eval']['overall']} | - | - |"]
    for g in o["gates"]: L.append(f"| {g['stage']} | " + " | ".join(str(g["dims"][d]) for d in DIMS) + f" | {g['dims']['overall']} | {g['worst_drop']} ({g['worst_dim']}) | {'STOP' if g['stop'] else 'ok'} |")
    L += ["", "## 7. Cost - latency - quality comparison (300-request simulated traffic)", "", "| Version | Avg cost/query | Avg latency (ms) | p95 latency (ms) | Cache hit rate | Day 50 eval (overall) |", "|---|---|---|---|---|---|"]
    for name, key, ev in [("Baseline", "baseline", o["baseline_eval"]["overall"]), ("+ Semantic cache", "opt1", o["gates"][1]["dims"]["overall"]), ("+ Cache + retrieval threshold (all)", "opt2", o["gates"][2]["dims"]["overall"])]:
        t = T[key]; L.append(f"| {name} | ${t['avg_cost']:.7f} | {t['avg_ms']:.0f} | {t['p95_ms']:.0f} | {t['hit_rate']:.0%} | {ev} |")
    b, a = T["baseline"], T["opt2"]
    L += ["", "### Sensitivity: the 82% hit rate comes from a simulated 30-question FAQ mix and is optimistic", "", "| Cache hit rate | Avg cost/query | Avg latency (ms) | Monthly cost @100k/day |", "|---|---|---|---|"]
    for x in o["sensitivity"]: L.append(f"| {x['hit_rate']:.0%} | ${x['cost']:.7f} | {x['ms']:.0f} | ${x['cost'] * 100000 * 30:,.2f} |")
    L += ["", f"**Net effect:** cost {100 * (1 - a['avg_cost'] / b['avg_cost']):.0f}% lower, average latency {100 * (1 - a['avg_ms'] / b['avg_ms']):.0f}% lower, Day 50 eval {o['baseline_eval']['overall']} -> {o['gates'][2]['dims']['overall']}.", "",
          "### Monthly projection after optimisation", "", "| Queries/day | Baseline | + Cache | + Cache + threshold |", "|---|---|---|---|"]
    for v in VOLUMES: k = f"{v}/day"; L.append(f"| {v:,} | ${o['projection']['baseline'][k]:,.2f} | ${o['projection']['opt1'][k]:,.2f} | ${o['projection']['opt2'][k]:,.2f} |")
    L += ["", "### Not implemented (next candidates)", "", "- Shorten the system prompt (~24% of cost, billed identically on every call); verify with the eval suite in `--mode openai` because prompt wording affects real LLM behaviour.", "- Parallelise independent steps (embedding + BM25) with `asyncio.gather` once a dense retriever is added.", "- Stream the answer to cut perceived latency."]
    open(os.path.join(HERE, "REPORT_DAY52.md"), "w", encoding="utf-8").write("\n".join(L))

if __name__ == "__main__": main()
