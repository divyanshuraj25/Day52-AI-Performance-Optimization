"""LLM judge (adapted from Day 29) with PayEasy-specific criteria.

Dimensions (each 0-1):
  correctness            - required facts present, no forbidden/false claims
  groundedness           - every claim supported by retrieved context (no invented facts)
  completeness           - ALL parts of the question answered
  refusal_appropriateness- abstains when KB has no answer; does NOT abstain when it does
Overall = 0.4*correctness + 0.2*groundedness + 0.2*completeness + 0.2*refusal_appropriateness
"""
import json, re
from rag_pipeline import tok, content_terms, REFUSAL

DIMS = ["correctness", "groundedness", "completeness", "refusal_appropriateness"]
WEIGHTS = {"correctness": 0.4, "groundedness": 0.2, "completeness": 0.2, "refusal_appropriateness": 0.2}

JUDGE_PROMPT = """You are a strict evaluator of a PayEasy payments support chatbot.
Question: {question}
Retrieved context: {context}
Chatbot answer: {answer}
Reference facts the answer must contain: {facts}
Should the bot refuse (no answer in KB)? {refuse}
Forbidden claims: {forbidden}

Score each dimension from 0.0 to 1.0:
- correctness: required facts present and nothing contradicting them or the forbidden claims.
- groundedness: every claim is supported by the retrieved context (arithmetic derived from context is OK).
- completeness: every part of the question is answered (e.g. both a price AND a saving).
- refusal_appropriateness: 1.0 if refused when it should, or answered when it should; 0.0 otherwise.
Return ONLY JSON: {{"correctness":x,"groundedness":x,"completeness":x,"refusal_appropriateness":x,"reason":"one sentence"}}"""

def _norm(s): return re.sub(r"\s+", " ", s.lower().replace(",", "")).strip()
def is_refusal(ans):
    a = _norm(ans)
    return any(p in a for p in ["don't have information", "do not have information", "cannot find", "not in my knowledge", "no information"])

def _fact_hit(fact, ans_n):
    return any(_norm(alt) in ans_n for alt in fact.split("|"))

def _overall(d): return round(sum(WEIGHTS[k] * d[k] for k in DIMS), 3)

def judge_offline(q, out):
    ans = out["answer"]; a = _norm(ans); refused = is_refusal(ans)
    facts = q.get("expected_facts", [])
    forb = [f for f in q.get("forbidden", []) if _norm(f) in a]
    forb += [p for p in q.get("forbidden_regex", []) if re.search(p, ans)]
    if q["should_refuse"]:
        d = dict(correctness=1.0 if refused else 0.0, groundedness=1.0 if refused else 0.0,
                 completeness=1.0 if refused else 0.0, refusal_appropriateness=1.0 if refused else 0.0)
        reason = "Correctly abstained." if refused else "Answered an unanswerable question instead of abstaining."
    else:
        hit = [f for f in facts if _fact_hit(f, a)]
        frac = len(hit) / len(facts) if facts else 1.0
        ctx = set(w for c in out["context"] for w in tok(c))
        sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if s.strip()]
        sup = [len(set(content_terms(s)) & ctx) / max(1, len(set(content_terms(s)))) >= 0.6 for s in sents]
        d = dict(correctness=0.0 if (forb or refused) else frac,
                 groundedness=0.0 if (forb or refused) else (sum(sup) / len(sup) if sup else 0.0),
                 completeness=0.0 if refused else frac,
                 refusal_appropriateness=0.0 if refused else 1.0)
        miss = [f for f in facts if f not in hit]
        reason = ("Wrongly refused. " if refused else "") + (f"Contains forbidden claim {forb}. " if forb else "") + (f"Missing facts: {miss}." if miss else "All required facts present.")
    d["overall"] = _overall(d); d["reason"] = reason
    return d

def judge_openai(q, out, model="gpt-4o-mini"):
    from openai import OpenAI
    p = JUDGE_PROMPT.format(question=q["question"], context=" | ".join(out["context"]), answer=out["answer"],
                            facts=q.get("expected_facts"), refuse=q["should_refuse"],
                            forbidden=(q.get("forbidden", []) + q.get("forbidden_regex", [])) or "none")
    r = OpenAI().chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                          messages=[{"role": "user", "content": p}])
    d = json.loads(r.choices[0].message.content)
    for k in DIMS: d[k] = float(d.get(k, 0))
    d["overall"] = _overall(d)
    return d

def judge(q, out, mode="offline"):
    return judge_openai(q, out) if mode == "openai" else judge_offline(q, out)
