"""Day 50 - Product-specific evaluation for the PayEasy support assistant.

Usage:
    python run_eval.py                 # offline mode (no API key, deterministic) - verifies the harness
    python run_eval.py --mode openai   # real LLM generation + real LLM judge (needs OPENAI_API_KEY)
Outputs: results/baseline_*.json, results/fixes_*.json, results/final_*.json, REPORT.md
"""
import argparse, json, os, statistics as st
from dataclasses import replace
from datetime import datetime
from rag_pipeline import BASELINE, FIXED, FIX_RETRIEVAL, FIX_PROMPT, FIX_GENERATION, answer
from llm_judge import judge, DIMS, is_refusal

TIERS = ["easy", "medium", "hard"]
HERE = os.path.dirname(os.path.abspath(__file__))

def load_dataset():
    with open(os.path.join(HERE, "eval_dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 30 and all(sum(q["tier"] == t for q in data) == 10 for t in TIERS)
    assert sum(1 for q in data if q.get("adversarial")) >= 3
    return data

def run_suite(data, cfg):
    rows = []
    for q in data:
        out = answer(q["question"], cfg)
        j = judge(q, out, cfg.mode)
        rows.append({"id": q["id"], "tier": q["tier"], "adversarial": q.get("adversarial"), "question": q["question"],
                     "answer": out["answer"], "retrieved": out["retrieved_ids"], "scores": {k: round(j[k], 3) for k in DIMS + ["overall"]},
                     "judge_reason": j.get("reason", "")})
    return rows

def tier_breakdown(rows):
    res = {t: round(st.mean(r["scores"]["overall"] for r in rows if r["tier"] == t), 3) for t in TIERS}
    res["overall"] = round(st.mean(r["scores"]["overall"] for r in rows), 3)
    return res

def dim_means(rows): return {d: round(st.mean(r["scores"][d] for r in rows), 3) for d in DIMS}

def diagnose(q, row):
    """Classify a failure as retrieval / prompt / generation and write ONE sentence."""
    need = set(q["source_chunks"]); got = set(row["retrieved"])
    if need and not need <= got:
        return "retrieval", (f"Retrieval: needed chunk(s) {sorted(need - got)} were not in the top-{len(got)} results (got {sorted(got)}), "
                             f"so the generator never saw the required facts.")
    if q["should_refuse"] and not is_refusal(row["answer"]):
        return "prompt", ("Prompt: the KB has no answer but the v1 prompt has no 'abstain if unsupported' rule, so the bot answered with an irrelevant "
                          "fact (hallucination risk).")
    return "generation", (f"Generation: the right context was retrieved but the answer is incomplete/wrong ({row['judge_reason'].strip()}) - "
                          f"answer was cut to a single sentence or needs arithmetic/multi-part reasoning.")

FIX_FOR = {"retrieval": ("F1 retrieval: top_k 1->3 + query expansion (synonyms)", FIX_RETRIEVAL),
           "prompt": ("F2 prompt: v2 system prompt with abstain/false-premise rules", FIX_PROMPT),
           "generation": ("F3 generation: answer budget 1->3 sentences / 'complete answer' instruction", FIX_GENERATION)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="offline", choices=["offline", "openai"])
    args = ap.parse_args(); mode = args.mode; ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = load_dataset(); byid = {q["id"]: q for q in data}
    base_cfg = replace(BASELINE, mode=mode); fixed_cfg = replace(FIXED, mode=mode)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    # 1) baseline
    base = run_suite(data, base_cfg); bt = tier_breakdown(base)
    worst_tier = min(TIERS, key=lambda t: bt[t])
    print(f"\n=== BASELINE ({mode}) ===\nTier scores: {bt}\nWeakest tier: {worst_tier}\nDimensions: {dim_means(base)}")

    # 2) five lowest + diagnosis
    worst5 = sorted(base, key=lambda r: (r["scores"]["overall"], r["id"]))[:5]
    diag = {}
    print("\n=== FIVE LOWEST-SCORING QUESTIONS ===")
    for r in worst5:
        kind, sentence = diagnose(byid[r["id"]], r); diag[r["id"]] = (kind, sentence)
        print(f"{r['id']} ({r['tier']}, {r['scores']['overall']}): {sentence}")

    # 3) apply the fix that matches each diagnosis, re-evaluate those questions
    fixes = []
    print("\n=== FIXES & RE-EVALUATION ===")
    for r in worst5:
        kind, _ = diag[r["id"]]; label, patch = FIX_FOR[kind]
        cfg = replace(base_cfg, name=f"fix_{kind}", **patch)
        out = answer(byid[r["id"]]["question"], cfg); j = judge(byid[r["id"]], out, mode)
        after = round(j["overall"], 3); before = r["scores"]["overall"]
        fixes.append({"id": r["id"], "failure_type": kind, "fix": label, "before": before, "after": after,
                      "improved": after > before, "answer_after": out["answer"]})
        print(f"{r['id']}: {label}\n     {before} -> {after}  {'IMPROVED' if after > before else 'NOT FIXED'}")
    n_fixed = sum(f["improved"] for f in fixes)
    print(f"\nFixed {n_fixed}/5 (requirement: >= 3)")

    # 4) full suite with all fixes combined -> check for regressions elsewhere
    final = run_suite(data, fixed_cfg); ft = tier_breakdown(final)
    print(f"\n=== ALL FIXES COMBINED ===\nTier scores: {ft}\nDimensions: {dim_means(final)}")
    regress = [(b["id"], b["scores"]["overall"], f["scores"]["overall"]) for b, f in zip(base, final) if f["scores"]["overall"] < b["scores"]["overall"]]
    print("Questions that got worse:", regress or "none")
    still = sorted(final, key=lambda r: r["scores"]["overall"])[:5]

    out = {"timestamp": ts, "mode": mode, "baseline": {"tiers": bt, "dimensions": dim_means(base), "rows": base},
           "diagnoses": {k: v[1] for k, v in diag.items()}, "fixes": fixes,
           "final": {"tiers": ft, "dimensions": dim_means(final), "rows": final}, "regressions": regress}
    path = os.path.join(HERE, "results", f"eval_{mode}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, ensure_ascii=False)
    write_report(out, worst_tier, still, byid)
    print(f"\nSaved {path} and REPORT.md")

def write_report(o, worst_tier, still, byid):
    L = [f"# Day 50 - Product-Specific Evaluation: PayEasy Support Assistant", "",
         f"Run mode: **{o['mode']}** | Timestamp: {o['timestamp']}", "",
         "> " + ("Offline mode = deterministic extractive generator + rule-based judge (harness verification, no API calls). "
                 "Re-run with `--mode openai` for real LLM numbers." if o["mode"] == "offline" else "Real OpenAI generation and LLM judge."), "",
         "## 1. Tier breakdown (overall judge score, 0-1)", "", "| Tier | Baseline | After fixes |", "|---|---|---|"]
    for t in TIERS + ["overall"]:
        L.append(f"| {t} | {o['baseline']['tiers'][t]} | {o['final']['tiers'][t]} |")
    L += ["", f"Weakest tier at baseline: **{worst_tier}**.", "", "## 2. Dimension scores", "", "| Dimension | Baseline | After fixes |", "|---|---|---|"]
    for d in DIMS: L.append(f"| {d} | {o['baseline']['dimensions'][d]} | {o['final']['dimensions'][d]} |")
    L += ["", "## 3. Judge scores per question (baseline -> fixed)", "", "| ID | Tier | Adversarial | Baseline | Fixed | Judge note (baseline) |", "|---|---|---|---|---|---|"]
    for b, f in zip(o["baseline"]["rows"], o["final"]["rows"]):
        L.append(f"| {b['id']} | {b['tier']} | {b['adversarial'] or ''} | {b['scores']['overall']} | {f['scores']['overall']} | {b['judge_reason']} |")
    L += ["", "## 4. Five failure diagnoses (baseline)", ""]
    for k, v in o["diagnoses"].items(): L.append(f"- **{k}** - {v}")
    L += ["", "## 5. Implemented fixes and re-evaluation", "", "| ID | Failure type | Fix | Before | After | Result |", "|---|---|---|---|---|---|"]
    for f in o["fixes"]: L.append(f"| {f['id']} | {f['failure_type']} | {f['fix']} | {f['before']} | {f['after']} | {'improved' if f['improved'] else 'not fixed'} |")
    L += ["", f"**{sum(f['improved'] for f in o['fixes'])}/5 fixed** (requirement: at least 3).", "",
          "## 6. Remaining weaknesses after fixes (lowest 5)", ""]
    for r in still: L.append(f"- {r['id']} ({r['scores']['overall']}): {r['judge_reason']}")
    if o["regressions"]: L += ["", "Questions that regressed after combined fixes: " + ", ".join(f"{i} ({a}->{b})" for i, a, b in o["regressions"])]
    L += ["", "## 7. Regression test", "", "`python regression_test.py` runs the full 30-question suite, writes a timestamped file to `results/` and prints PASS/FAIL."]
    open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L))

if __name__ == "__main__":
    main()
