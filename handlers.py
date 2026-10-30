"""Edge-case handlers + safe_answer(): the production entry point.

Every handler returns a HELPFUL message that tells the user what to do - never a stack trace.
"""
import logging, re, traceback, uuid
from dataclasses import replace
from normalize import normalize_with_notes
from rag_pipeline import FIXED, answer, tok

logging.basicConfig(filename="errors.log", level=logging.ERROR, format="%(asctime)s %(message)s")
RAW_LIMIT = 20000      # reject anything longer than this before doing any work
EXAMPLES = "For example: 'How long is the refund window?' or 'What does the Growth plan cost?'"

MESSAGES = {
    "empty_input": "I didn't catch a question. Please type a question about PayEasy. " + EXAMPLES,
    "no_text": "I couldn't find any readable words in your message. Please rephrase it as a short question about PayEasy. " + EXAMPLES,
    "too_long": "Your message is too long for me to process (limit is {limit:,} characters). Please shorten it to one or two sentences and send it again.",
    "prompt_injection": "I can't follow instructions that try to change my rules or reveal internal settings. I can help with PayEasy questions about plans, fees, payouts, refunds, security or the API - please ask one of those.",
    "unsafe_input": "That message contains code or markup that I can't process. Please type your question in plain words. " + EXAMPLES,
    "non_english": "I can currently answer in English only. Please rewrite your question in English (for example: 'How much does the Growth plan cost per month?') or email support, which is available 24/7.",
    "out_of_scope": "That's outside what I can help with - I only answer questions about PayEasy (plans and pricing, fees, payouts, refunds, account and security, the API). {hint}You can ask me something like: 'What are PayEasy's processing fees?'",
    "internal_error": "Sorry, something went wrong on my side while answering. Please try again in a moment or rephrase your question. If it keeps happening, email support and quote reference {ref}.",
}
OOS_HINTS = [(("code", "python", "program", "script", "javascript"), "For coding help, try a coding assistant or the official language docs. "),
             (("tax", "irs", "income"), "For tax questions, a qualified tax professional is the right place. "),
             (("stock", "invest", "shares", "crypto"), "For investment decisions, please speak to a licensed financial adviser. "),
             (("weather", "rain", "temperature"), "For weather, a weather service or app will be more accurate. "),
             (("laptop", "phone", "buy", "recommend"), "For product recommendations, review sites are a better source. "),
             (("capital", "who won", "cricket", "history"), "A general search engine is best for general-knowledge questions. ")]

INJECTION = re.compile(r"ignore (all |any )?(the )?(previous|prior|above)|disregard (all |your )?(previous|prior|instructions)|system prompt|"
                       r"reveal (your|the) (instructions|prompt|rules)|you are now|act as (an? )?(dan|admin|root)|jailbreak|developer mode|"
                       r"admin password|override (your )?(rules|safety)", re.I)
UNSAFE = re.compile(r"\b(drop|truncate)\s+table\b|\bunion\s+select\b|\bselect\s+.+\s+from\b|\bor\s+1\s*=\s*1\b|;\s*--|`.*`|\$\(.*\)|\|\s*(bash|sh)\b|\.\./", re.I)

FOREIGN_WORDS = set("cuanto cuánto cuesta qué como cómo el la los las al mes por para una un es de del con puedo quiero precio plan quel combien est le les des du pour wie viel kostet der die das ist und für ich kann".split()) - {"plan", "un", "es", "de", "la", "est", "wie"}

DOMAIN_TERMS = set(tok(w)[0] for w in """refund price pricing plan starter growth scale fee payout transaction chargeback dispute invoice invoicing password
login account two-factor 2fa api webhook trial subscription billing payment card currency bank security pci support seat upgrade downgrade
country countries business charge rate key paid cost discount annual monthly pay payeasy payer merchant checkout team member""".split())

def is_non_english(t):
    letters = [c for c in t if c.isalpha()]
    if not letters: return False
    if sum(ord(c) > 0x024F for c in letters) / len(letters) > 0.3: return True      # Devanagari, Arabic, CJK, Cyrillic...
    if "¿" in t or "¡" in t: return True
    return len(set(re.findall(r"[a-záéíóúñüàèçäößê]+", t.lower())) & FOREIGN_WORDS) >= 2

def in_scope(t): return any(w in DOMAIN_TERMS for w in tok(t))

def oos_message(t):
    q = t.lower(); hint = ""
    for keys, h in OOS_HINTS:
        if any(k in q for k in keys): hint = h; break
    return MESSAGES["out_of_scope"].format(hint=hint)

def _reply(handler, text, **kw):
    return {"answer": text, "retrieved_ids": [], "context": [], "handler": handler, **kw}

def precheck(query):
    """Normalise + run all cheap local guards. Returns (text, handler_name | None, message | None, notes)."""
    if isinstance(query, str) and len(query) > RAW_LIMIT:
        return "", "too_long", MESSAGES["too_long"].format(limit=RAW_LIMIT), []
    n = normalize_with_notes(query); t = n.text
    if not t: return t, "empty_input", MESSAGES["empty_input"], n.notes
    if INJECTION.search(t): return t, "prompt_injection", MESSAGES["prompt_injection"], n.notes
    if UNSAFE.search(t): return t, "unsafe_input", MESSAGES["unsafe_input"], n.notes
    if not re.search(r"[A-Za-z]", t) and not is_non_english(t): return t, "no_text", MESSAGES["no_text"], n.notes
    if is_non_english(t): return t, "non_english", MESSAGES["non_english"], n.notes
    if not in_scope(t): return t, "out_of_scope", oos_message(t), n.notes
    return t, None, None, n.notes

def safe_answer(query, cfg=FIXED):
    """Never raises. Returns dict with answer, context, retrieved_ids, handler (None if normal path)."""
    try:
        t, handler, msg, notes = precheck(query)
        if handler: return _reply(handler, msg)
        out = answer(t, cfg)
        if not out.get("answer", "").strip(): return _reply("internal_error", MESSAGES["internal_error"].format(ref="EMPTY"))
        out["handler"] = None; out["normalisation"] = notes
        return out
    except Exception:
        ref = uuid.uuid4().hex[:8]; logging.error("ref=%s query=%r\n%s", ref, str(query)[:200], traceback.format_exc())
        return _reply("internal_error", MESSAGES["internal_error"].format(ref=ref))
