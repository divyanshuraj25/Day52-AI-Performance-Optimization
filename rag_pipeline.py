"""PayEasy support assistant: retrieval + generation.

Two generation backends:
  * mode="openai"  -> real LLM (needs OPENAI_API_KEY)
  * mode="offline" -> deterministic extractive generator (no network; used to verify the harness)
Retrieval is BM25 in both modes.
"""
import math, os, re
from collections import Counter
from dataclasses import dataclass, replace
from knowledge_base import CHUNKS

STOP = set("""a an the is are was were be been do does did i me my we our you your it its of to in on for with and or but
if as at by from that this these those can could should would will what which who how when where why there their then than so
any all some about into out up down over under per have has had get got need want use using run am not no yes please
payeasy pay easy""".split())

SYNONYMS = {  # query expansion: user vocabulary -> knowledge-base vocabulary
    "cost": "price pricing plan", "much": "price", "charge": "fee", "charged": "fee", "dispute": "chargeback",
    "disputes": "chargeback", "disputed": "chargeback", "money back": "refund", "cheapest": "price",
    "paid out": "payout", "get paid": "payout", "pay out": "payout", "daily": "business day payout",
    "team members": "user seat", "members": "user seat", "annual": "annual billing discount",
    "downgrade": "downgrade billing cycle", "upgrade": "upgrade", "trial": "free trial",
    "sign up": "countries supported waitlist", "business in": "countries supported waitlist",
    "total fee": "processing fee", "pci": "PCI DSS", "compliant": "certified",
    "germany": "countries supported waitlist", "india": "countries supported waitlist", "france": "countries supported waitlist",
    "japan": "countries supported waitlist", "brazil": "countries supported waitlist",
}
COV_STOP = set("number total much many long time year month day days week should people someone checkout team sign up work works".split())

def tok(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 4 and w.endswith("sses"): w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]
        out.append(w)
    return out

def content_terms(text):
    return [w for w in tok(text) if w not in STOP]

@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    mode: str = "offline"
    top_k: int = 1
    max_sentences: int = 1
    expand_queries: bool = False
    abstain_guard: bool = False        # offline: abstain on out-of-scope topics; openai: handled by prompt v2
    prompt_version: str = "v1"
    min_rel_score: float = 0.0         # keep only chunks scoring >= min_rel_score * best score (0 = keep top_k)
    model: str = "gpt-4o-mini"

BASELINE = Config()
# Fixes are applied cumulatively; each addresses one failure class (see run_eval.py)
FIX_RETRIEVAL = dict(top_k=3, expand_queries=True)
FIX_PROMPT = dict(abstain_guard=True, prompt_version="v2")
FIX_GENERATION = dict(max_sentences=3)
FIXED = replace(BASELINE, name="fixed", **FIX_RETRIEVAL, **FIX_PROMPT, **FIX_GENERATION)

# ---------------- retrieval ----------------
_docs = [(c, tok(c["title"] + " " + c["text"])) for c in CHUNKS]
_N = len(_docs)
_avgdl = sum(len(t) for _, t in _docs) / _N
_df = Counter(w for _, t in _docs for w in set(t))
def _idf(w): return math.log(1 + (_N - _df.get(w, 0) + 0.5) / (_df.get(w, 0) + 0.5))

def expand(query):
    q = query.lower(); extra = []
    for k, v in SYNONYMS.items():
        if k in q: extra.append(v)
    return query + " " + " ".join(extra)

def retrieve(query, cfg):
    q = expand(query) if cfg.expand_queries else query
    qt = content_terms(q); scores = []
    for c, t in _docs:
        tf = Counter(t); s = 0.0
        for w in qt:
            if w in tf:
                s += _idf(w) * tf[w] * 2.5 / (tf[w] + 1.5 * (0.25 + 0.75 * len(t) / _avgdl))
        scores.append((c, s))
    scores.sort(key=lambda x: -x[1])
    top = scores[:cfg.top_k]
    if cfg.min_rel_score and top and top[0][1] > 0:
        top = [(c, s) for c, s in top if s >= cfg.min_rel_score * top[0][1]] or top[:1]
    return top

OUT_OF_SCOPE = ["crypto", "bitcoin", "ethereum", "apple pay", "google pay", "paypal", "venmo", "stripe", "cash app", "klarna", "afterpay", "zelle"]
def out_of_scope(query):
    """Offline stand-in for the v2 prompt's 'abstain if the context lacks the answer' rule."""
    q = query.lower()
    return any(t in q for t in OUT_OF_SCOPE)

# ---------------- generation ----------------
REFUSAL = "I don't have information about that in my knowledge base."
SYSTEM_V1 = "You are PayEasy's support assistant. Answer the user's question using the context."
SYSTEM_V2 = ("You are PayEasy's support assistant. Answer ONLY from the context. If the context does not contain the answer, "
             "reply exactly: \"I don't have information about that in my knowledge base.\" Never invent numbers, phone numbers or features. "
             "If the question contains a false premise, correct it using the context. For calculations, show the arithmetic step by step "
             "and give the final figure. Give a complete answer covering every part of the question.")

def _split(text): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _offline_generate(query, retrieved, cfg):
    qt = set(content_terms(expand(query) if cfg.expand_queries else query))
    sents = []
    for c, _ in retrieved:
        for s in _split(c["text"]):
            sents.append((len(qt & set(content_terms(s))) / (1 + 0.05 * len(tok(s))), s))
    sents.sort(key=lambda x: -x[0])
    return " ".join(s for _, s in sents[:cfg.max_sentences])

def _openai_generate(query, retrieved, cfg):
    from openai import OpenAI
    client = OpenAI()
    ctx = "\n".join(f"[{c['id']}] {c['text']}" for c, _ in retrieved)
    system = SYSTEM_V2 if cfg.prompt_version == "v2" else SYSTEM_V1
    r = client.chat.completions.create(model=cfg.model, temperature=0, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}])
    return r.choices[0].message.content.strip()

def answer(query, cfg=BASELINE):
    retrieved = retrieve(query, cfg)
    if cfg.mode == "offline" and cfg.abstain_guard and out_of_scope(query):
        text = REFUSAL                    # offline stand-in for the v2 prompt's abstain instruction
    elif cfg.mode == "openai":
        text = _openai_generate(query, retrieved, cfg)
    else:
        text = _offline_generate(query, retrieved, cfg)
    return {"answer": text, "retrieved_ids": [c["id"] for c, _ in retrieved], "context": [c["text"] for c, _ in retrieved]}
