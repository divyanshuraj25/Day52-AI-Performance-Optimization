"""Input normalisation - runs BEFORE any other processing.

Does: type coercion, Unicode NFKC (fullwidth -> ASCII), removal of zero-width/control chars,
Cyrillic/Greek lookalike -> Latin repair (only inside mixed-script words), HTML/script stripping,
whitespace strip + collapse, collapse of runaway repeated words, length cap.
"""
import re, unicodedata
from dataclasses import dataclass, field

MAX_CHARS = 1000          # hard cap on text passed to retrieval / the LLM

CONFUSABLES = {  # lookalike -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ɡ": "g",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "І": "I", "М": "M", "Н": "H", "К": "K", "Т": "T", "В": "B",
    "ο": "o", "α": "a", "ν": "v", "ι": "i", "ρ": "p", "Ο": "O", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Τ": "T",
}
_HTML_BLOCK = re.compile(r"<\s*(script|style)[^>]*>.*?<\s*/\s*\1\s*>", re.I | re.S)
_HTML_TAG = re.compile(r"<[^>]{0,200}>")
_REPEAT = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.I)

@dataclass
class Normalized:
    text: str
    notes: list = field(default_factory=list)

def _fix_lookalikes(word):
    has_latin = any("a" <= c.lower() <= "z" for c in word)
    if has_latin and any(c in CONFUSABLES for c in word):
        return "".join(CONFUSABLES.get(c, c) for c in word)
    return word

def normalize_with_notes(text, max_chars=MAX_CHARS):
    notes = []
    if text is None:
        text = ""; notes.append("none_to_empty")
    elif isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", "replace"); notes.append("decoded_bytes")
    elif not isinstance(text, str):
        text = str(text); notes.append("coerced_non_string")
    t = unicodedata.normalize("NFKC", text)
    if t != text: notes.append("unicode_nfkc")
    t = re.sub(r"[\t\n\r\v\f]", " ", t)
    cleaned = "".join("" if unicodedata.category(c) in ("Cf", "Cc", "Co", "Cs") else c for c in t)
    if cleaned != t: notes.append("removed_invisible_chars")
    t = cleaned
    t2 = _HTML_TAG.sub(" ", _HTML_BLOCK.sub(" ", t))
    if t2 != t: notes.append("stripped_html")
    t = t2
    t2 = " ".join(_fix_lookalikes(w) for w in t.split(" "))
    if t2 != t: notes.append("fixed_lookalike_chars")
    t = t2
    t = re.sub(r"\s+", " ", t).strip()
    t2 = _REPEAT.sub(r"\1", t)
    if t2 != t: notes.append("collapsed_repeated_words")
    t = t2
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0]; notes.append("truncated")
    return Normalized(t, notes)

def normalize_input(text):
    """Public helper: returns the normalised string."""
    return normalize_with_notes(text).text
